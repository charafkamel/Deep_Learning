import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from helpers import user_login
from logger import CustomLogger
from data_processing import load_dataset_from_disk


class SFTGenerativeTrainerBase:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", config=None, logger=None):
        self.config = config
        self.logger = logger
        self.model_name = model_name
        self.load_model_and_tokenizer(model_name)
        self.preprocess_and_split_dataset()
        self.setup_training_args_and_trainer()

    # 1) Load model and tokenizer
    def load_model_and_tokenizer(self, model_name):
        if self.logger:
            self.logger.info("Loading model and tokenizer")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        # Ensure proper padding and EOS handling
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    # 2) Format data for causal LM
    def preprocess_and_split_dataset(self):
        if self.logger:
            self.logger.info("Loading and preprocessing dataset")

        self.config["dataset_path"] = os.path.join(os.getcwd(), "cleaned_data", "sft_dataset")
        dataset = load_from_disk(self.config["dataset_path"])

        def _preprocess(example):
            prompt = f"detoxify: {example['toxic']}\nNeutral:"
            full_text = f"{prompt} {example['neutral']}"
            return {"text": full_text}

        tokenized_dataset = dataset.map(_preprocess, remove_columns=dataset.column_names)
        split = tokenized_dataset.train_test_split(test_size=0.1)
        self.train_dataset, self.eval_dataset = split["train"], split["test"]

    # 3) SFTTrainer with SFTTrainingArguments
    def setup_training_args_and_trainer(self):
        if self.logger:
            self.logger.info("Setting up training arguments and SFTTrainer")
        self.model_name_for_hub = self.model_name.split("/")[-1]
        self.new_model_name = f"TarhanE/sft-base_loss-{self.model_name_for_hub}-mle{0}-ul{0}-tox{0}-e{self.config['sft_params_generative_base']['num_train_epochs']}"
        self.training_args = SFTConfig(**self.config["sft_params_generative_base"], hub_model_id=self.new_model_name,
            run_name=  self.new_model_name)

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

    # 4) Train with checkpointing and save
    def train(self, resume_from_checkpoint=False):
        if self.logger:
            self.logger.info("Starting training")

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        final_epoch = int(self.trainer.state.epoch or 0)
        out_dir = os.path.join(self.trainer.args.output_dir, f"e{final_epoch}")
        os.makedirs(out_dir, exist_ok=True)

        self.trainer.save_model(out_dir)
        self.tokenizer.save_pretrained(out_dir)

        if self.logger:
            self.logger.info(f"Model & tokenizer saved to {out_dir}")
        self.trainer.push_to_hub()
        return out_dir
