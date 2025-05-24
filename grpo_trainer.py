import os
import logging
import yaml
from custom_loss import SimilarityToxicityReward
from helpers import user_login
from logger import CustomLogger
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import Dataset
from datasets import load_dataset
from data_processing import load_dataset_from_disk
from transformers import DataCollatorForSeq2Seq
import torch


def load_config(config_path="main_config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class CustomGRPOTrainer:
    def __init__(
        self,
        policy_model_name: str,
        df_dict,
        config
    ):
        # Store args
        self.df_dict = df_dict
        self.model_name = policy_model_name
        # GRPO config
        self.grpo_config = GRPOConfig(
            **config["grpo_params"],
        )

        # Initialize components
        self._init_tokenizers_and_models()
        self._init_datasets()
        self._init_reward_model()
        self._init_data_collator()
        self._init_grpo_trainer()

    def _init_tokenizers_and_models(self):
        # Policy & reference
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.policy_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_name).eval()
        

    def _init_datasets(self):
        ### Load the dataset from disk the rl_dataset
        self.config["dataset_path"] = os.path.join(os.getcwd(), "cleaned_data", "rl_dataset")


    def _init_reward_model(self):
        self.reward_model = SimilarityToxicityReward(
            w_sim=1.0,
            w_tox=1.0,
        )
    
    def _reward_fn(self, samples, prompts, model_inputs=None, **kwargs):
        # Reward from Similarity + (1 - Toxicity)
        with torch.no_grad():
            rewards = self.reward_model(prompts, samples).detach()  # shape (B,)
        
        # KL penalty
        logprobs, ref_logprobs = trl_utils.logprobs_of_sequences(
            self.policy_model, self.tokenizer,
            samples, model_inputs,
            ref_model=self.ref_model
        )
        kl_vals = (logprobs - ref_logprobs).sum(dim=-1).detach()
        clipped_kl = kl_vals.clamp(min=0.0, max=2.0)
        # Final reward = reward - lambda * KL
        _lambda = 0.1
        final_reward = rewards - _lambda * clipped_kl
        return final_reward.tolist()


    def _init_grpo_trainer(self):
        self.trainer = GRPOTrainer(
            model=self.policy_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.rl_dataset,
            reward_fn=self._reward_fn,
            config=self.grpo_config,
        )

    def train(self, resume_from_checkpoint=False):
        # Run GRPO training
        self.trainer.train()
        # Save outputs
        os.makedirs(self.output_dir, exist_ok=True)
        self.policy_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Saved fine-tuned model to {self.output_dir}")