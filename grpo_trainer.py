import os
import logging
import yaml
from custom_loss import SimilarityToxicityReward
from helpers import user_login
from logger import CustomLogger
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import Dataset
from datasets import load_dataset
from data_processing import load_dataset_from_disk
import torch
from datasets import load_from_disk


def load_config(config_path="main_config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class CustomGRPOTrainer:
    def __init__(
        self,
        policy_model_name: str,
        config,
        logger,
    ):
        self.model_name = policy_model_name
        self.config = config
        self.logger = logger

        self.model_name_for_hub = policy_model_name.split("/")[-1]
        self.new_model_name = f"TarhanE/GRPO-{self.model_name_for_hub}"
        self.output_dir = config.get("output_dir", "./grpo_results")

        self.grpo_config = GRPOConfig(
            **config["grpo_params"],
            hub_model_id=self.new_model_name,
            run_name=self.new_model_name,
        )

        self.logger.info("Initializing tokenizers and models")
        self._init_tokenizers_and_models()
        self.logger.info("Initializing datasets")
        self._init_datasets()
        self.logger.info("Initializing reward model")
        self._init_reward_model()
        self.logger.info("Initializing trainer")
        self._init_grpo_trainer()
        self.logger.info("Trainer initialized")

    def _init_tokenizers_and_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get the device to use
        # Check if CUDA is available and set device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.policy_model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_name).eval().to(self.device) # Ensure ref_model is also on device

    def _init_datasets(self):
        self.rl_dataset = load_from_disk(os.path.join(os.getcwd(), "cleaned_data", "rl_dataset"))
        self.eval_dataset = load_from_disk(os.path.join(os.getcwd(), "cleaned_data", "eval_dataset"))
        self.rl_dataset = self.rl_dataset.map(lambda x: {
            "prompt": x["toxic"],
            "output": x["neutral"],
        })
        self.eval_dataset = self.eval_dataset.map(lambda x: {
            "prompt": x["toxic"],
            "output": x["neutral"],
        })

    def _init_reward_model(self):
        self.reward_model = SimilarityToxicityReward(
            w_sim=1.0,
            w_tox=1.0,
        )
        # Assuming SimilarityToxicityReward might also contain models, move it to device
        # If it's a simple function, this won't apply. If it wraps a SentenceTransformer or similar,
        # you'll need to ensure that model is also moved.
        # Example if it has a .to(device) method:
        # self.reward_model.to(self.device)


    def _reward_fn(
        self,
        prompts: list[str],
        completions: list[str],
        model_inputs=None,
        **kwargs,
    ):
        def _get_log_probs(model, tokenizer, texts):
            # The .to(model.device) ensures inputs are on the same device as the model
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()} # Crucial for moving inputs
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = inputs['input_ids'][:, 1:].contiguous()
            log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
            log_probs_of_tokens = torch.gather(log_probs, -1, shifted_labels.unsqueeze(-1)).squeeze(-1)
            attention_mask_shifted = inputs['attention_mask'][:, 1:]
            summed_log_probs = (log_probs_of_tokens * attention_mask_shifted).sum(dim=-1)
            return summed_log_probs

        with torch.no_grad():
            # Ensure reward_model is also on the correct device if it contains torch components
            task_reward = self.reward_model(prompts, completions).detach()

        policy_logprobs_sum = _get_log_probs(self.policy_model, self.tokenizer, completions)
        ref_logprobs_sum = _get_log_probs(self.ref_model, self.tokenizer, completions)

        # These should now be on the same device if the models are on the same device
        kl_vals = (policy_logprobs_sum - ref_logprobs_sum).detach()
        kl_pen = kl_vals.clamp(min=0.0, max=2.0)

        _lambda = 0.1
        final_reward = task_reward - _lambda * kl_pen
        self.logger.debug(f"Task reward: {task_reward.tolist()}")
        self.logger.debug(f"Final reward: {final_reward.tolist()}")
        return final_reward.tolist()

    def _init_grpo_trainer(self):
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        self.trainer = GRPOTrainer(
            model=self.policy_model,
            args=self.grpo_config,
            train_dataset=self.rl_dataset,
            eval_dataset=self.eval_dataset,
            reward_funcs=self._reward_fn
        )

    def train(self, resume_from_checkpoint=False):
        self.trainer.train()
        os.makedirs(self.output_dir, exist_ok=True)
        self.policy_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Saved fine-tuned model to {self.output_dir}")