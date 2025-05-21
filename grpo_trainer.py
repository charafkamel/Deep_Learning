import os
import logging
import yaml
from custom_loss import SimilarityToxicityLoss
# from helpers import user_login # Not used in this snippet
from logger import CustomLogger
# GRPO specific imports
from trl import GRPOTrainer, GRPOConfig, create_reference_model, AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, GenerationConfig
from datasets import load_from_disk
import torch
from torch import nn


def load_config(config_path="main_config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


class SimilarityToxicityRewardModel(SimilarityToxicityLoss, nn.Module):
    """Wraps SimilarityToxicityLoss as a HF-style reward model for GRPOTrainer."""
    def __init__(self, config, w_sim, w_tox, tokenizer, device, logger=None):
        SimilarityToxicityLoss.__init__(
            self, config=config, w_sim=w_sim, w_tox=w_tox, device=device, logger=logger
        )
        nn.Module.__init__(self)
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, input_ids, attention_mask=None):
        # Decode sequences to full text (prompt + response)
        texts = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
        prompts, responses = [], []
        eos = self.tokenizer.eos_token or self.tokenizer.sep_token
        for txt in texts:
            if eos and eos in txt:
                pre, post = txt.split(eos, 1)
                prompts.append(pre)
                responses.append(post)
            else:
                prompts.append(txt) # If no EOS, the whole text is treated as prompt
                responses.append("") # And response is empty
        
        # Compute loss (lower is better) using the SimilarityToxicityLoss logic
        loss_vals = super().forward(prompts, responses)  # tensor (B,)
        if not isinstance(loss_vals, torch.Tensor):
            loss_vals = torch.tensor(loss_vals, device=self.device)
        loss_vals = loss_vals.detach().to(self.device)
        
        # Convert to reward: higher reward for lower loss
        rewards = -loss_vals
        
        # Return object with `.logits` shape (B,1) as expected by trl trainers
        return type("RewardOut", (), {"logits": rewards.unsqueeze(1)})()
    

class CustomSeq2SeqRLTrainer:
    def __init__(
        self,
        config,
        device: str = "cuda",
        logger: logging.Logger = None,
    ):
        # Load config and set up
        self.config = config
        self.device = device
        self.logger = logger

        # GRPO config
        self.grpo_config = GRPOConfig(**self.config["grpo_params"]) # Use GRPOConfig

        if self.logger:
            self.logger.info("Initializing Seq2Seq GRPO trainer...")

        self._init_tokenizer_and_models()
        self._init_loss_fn() # Still used by the reward model internally
        self._init_reward_model()
        self._init_datasets_and_collator()
        self._init_grpo_trainer() # Call the GRPO trainer initialization

    def _init_tokenizer_and_models(self):
        # Load policy model (with value head) and reference model
        base_name = os.path.join(self.config["sft_params"]["output_dir"], "checkpoint-6200")
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        # This is critical: Use AutoModelForSeq2SeqLMWithValueHead for Seq2Seq GRPO
        self.policy_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(base_name).to(self.device)
        
        # Set up generation config for the policy model
        try:
            gen_conf = GenerationConfig.from_pretrained(base_name)
        except Exception:
            # Fallback if config.json does not contain generation config
            gen_conf = GenerationConfig(**self.policy_model.config.to_dict())
        self.policy_model.generation_config = gen_conf
        
        # Create a frozen reference model for KL divergence calculation
        self.ref_model = create_reference_model(self.policy_model).eval().to(self.device)

    def _init_loss_fn(self):
        # This loss function is used internally by the SimilarityToxicityRewardModel
        self.loss_fn = SimilarityToxicityLoss(
            config=self.config,
            w_sim=1.0, # Weights passed from main config or default
            w_tox=1.0,
            device=self.device,
            logger=self.logger,
        )

    def _init_reward_model(self):
        # Your custom reward model, which is a wrapper around SimilarityToxicityLoss
        self.reward_model = SimilarityToxicityRewardModel(
            config=self.config,
            w_sim=self.config.get("w_sim", 1.0),
            w_tox=self.config.get("w_tox", 1.0),
            tokenizer=self.tokenizer,
            device=self.device,
            logger=self.logger,
        ).to(self.device).eval() # Ensure it's on the correct device and in eval mode

    def _init_datasets_and_collator(self):
        rl_path = os.path.join(os.getcwd(), "cleaned_data", "rl_dataset")
        eval_path = os.path.join(os.getcwd(), "cleaned_data", "eval_dataset")
        self.rl_dataset = load_from_disk(rl_path)
        self.eval_dataset = load_from_disk(eval_path)

        # Map 'toxic' column to 'prompt' as expected by trl trainers
        self.rl_dataset = self.rl_dataset.map(lambda ex: {"prompt": ex["toxic"]})
        self.eval_dataset = self.eval_dataset.map(lambda ex: {"prompt": ex["toxic"]})

        # Data collator for Seq2Seq models
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.policy_model, # Provides model-specific padding tokens
            padding=True,
            return_tensors="pt"
        )

    def _init_grpo_trainer(self):
        self.grpo_trainer = GRPOTrainer( # Use GRPOTrainer
            args=self.grpo_config,
            model=self.policy_model,
            reward_model=self.reward_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer, # Pass tokenizer for generation and tokenization
            train_dataset=self.rl_dataset,
            data_collator=self.data_collator,
        )

    def train(self, epochs: int = 1):
        for epoch in range(epochs):
            self.logger.info(f"Starting GRPO epoch {epoch + 1}/{epochs}")
            stats = self.grpo_trainer.train() # Call train method on GRPOTrainer
            self.logger.info(f"Epoch {epoch + 1} stats: {stats}")

        # Save the final policy model and tokenizer
        # Ensure the output directory is configured in grpo_params
        out_dir = self.config["grpo_params"]["output_dir"] 
        os.makedirs(out_dir, exist_ok=True)
        self.policy_model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        self.logger.info(f"Saved fine-tuned model to {out_dir}")