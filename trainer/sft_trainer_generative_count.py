import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)
from datasets import load_from_disk
from utils.logger import CustomLogger


# ################################################################### #
# COUNTLossTrainerGenerative for generative models with toxicity loss #
# ################################################################### #
class COUNTLossTrainerGenerative(Trainer):
    def __init__(self, *args, tokenizer=None, tox_tokenizer=None, tox_model=None, loss_weights=None, log_interval=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.tox_tokenizer = tox_tokenizer
        self.tox_model = tox_model
        self.loss_weights = loss_weights or {"mle": 0.5, "ul": 0.5, "tox": 1.0}
        self.log_interval = log_interval
        self._step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        tox_dev = next(self.tox_model.parameters()).device
        if tox_dev != model.device:
            self.tox_model.to(model.device)

        self._step_count += 1

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        labels = inputs["labels"].to(model.device)

        # 1. MLE loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_loss = outputs.loss

        # 2. Unlikelihood loss
        with torch.no_grad():
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            neg_log_probs = torch.log(1.0 - probs + 1e-6)

            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            safe_labels = safe_labels.to(model.device)

            ul = -neg_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            pad_mask = labels != -100
            ul_loss = (ul * pad_mask).sum() / pad_mask.sum()

        # 3. Toxicity penalty
        with torch.no_grad():
            gen_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256)

            gen_ids = torch.where(gen_ids >= self.tokenizer.vocab_size,
                                  torch.tensor(self.tokenizer.pad_token_id, device=gen_ids.device),
                                  gen_ids)

            gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            tox_inputs = self.tox_tokenizer(gen_texts, return_tensors="pt", truncation=True, padding=True)
            tox_inputs = {k: v.to(model.device) for k, v in tox_inputs.items()}
            tox_logits = self.tox_model(**tox_inputs).logits
            tox_probs = torch.sigmoid(tox_logits)
            tox_scores = tox_probs[:, 0]
            tox_loss = tox_scores.mean()

        total_loss = (
            self.loss_weights["mle"] * mle_loss +
            self.loss_weights["ul"] * ul_loss +
            self.loss_weights["tox"] * tox_loss
        )

        if self._step_count % self.log_interval == 0:
            print(f"[Step {self._step_count}] MLE: {mle_loss.item():.4f} | COUNT: {ul_loss.item():.4f} | Toxicity: {tox_loss.item():.4f} | Total: {total_loss.item():.4f}")

        return (total_loss, outputs) if return_outputs else total_loss


class CustomGenerativeCountTrainer:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", tox_model_name="unitary/toxic-bert", config=None, logger=None):
        self.config = config or {}
        self.logger = logger or CustomLogger(__name__)
        self.repo_id = self.config.get("hf_username", "kamelcharaf")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # needed for decoder-only models

        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.tox_tokenizer = AutoTokenizer.from_pretrained(tox_model_name)
        self.tox_model = AutoModelForSequenceClassification.from_pretrained(tox_model_name).eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.tox_model.to(device)

        dataset_path = self.config.get("dataset_path", os.path.join(os.getcwd(), "cleaned_data", "sft_dataset"))
        dataset = load_from_disk(dataset_path)

        def preprocess(example):
            input_text = "detoxify: " + example["toxic"]
            target_text = example["neutral"]
            full_text = input_text + " " + target_text
            enc = self.tokenizer(full_text, max_length=128, padding="max_length", truncation=True)
            enc["labels"] = [t if t != self.tokenizer.pad_token_id else -100 for t in enc["input_ids"]]
            return enc

        tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)
        split = tokenized.train_test_split(test_size=self.config.get("eval_split", 0.1), seed=self.config.get("seed", 42))
        self.train_dataset = split["train"]
        self.eval_dataset = split["test"]

        count_params = self.config.get("sft_params_dcount_generative", {})

        self.loss_weights = {
            "mle": self.config.get("mle_weight", 0.5),
            "ul": self.config.get("ul_weight", 0.5),
            "tox": self.config.get("tox_weight", 1.0)
        }

        self.model_name_for_hub = model_name.split("/")[-1].replace("/", "-")
        self.new_model_name = f"{self.repo_id}/sft-count_loss-{self.model_name_for_hub}-mle{self.loss_weights['mle']}-ul{self.loss_weights['ul']}-tox{self.loss_weights['tox']}-e{self.config['sft_params_dcount_generative']['num_train_epochs']}"

        self.training_args = TrainingArguments(
            **count_params,
            hub_model_id=self.new_model_name,
            run_name=self.new_model_name
        )

        self.trainer = COUNTLossTrainerGenerative(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            tokenizer=self.tokenizer,
            tox_tokenizer=self.tox_tokenizer,
            tox_model=self.tox_model,
            loss_weights=self.loss_weights,
            log_interval=self.config.get("log_interval", 50)
        )

    def train(self, resume_from_checkpoint=False):
        self.logger.info("[CountTrainer] Starting training")
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        epoch = int(self.trainer.state.epoch or 0)
        out_dir = os.path.join(self.training_args.output_dir, f"count_e{epoch}")
        os.makedirs(out_dir, exist_ok=True)
        self.trainer.save_model(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        self.logger.info(f"[CountTrainer] Model & tokenizer saved to {out_dir}")
        self.trainer.push_to_hub(
            repo_id=self.new_model_name,
            commit_message="Pushing model to Hugging Face Hub",
            blocking=True,
        )
        return out_dir