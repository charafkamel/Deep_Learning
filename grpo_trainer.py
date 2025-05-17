import os
import logging
import yaml
from helpers import user_login
from logger import CustomLogger

def load_config(config_path="main_config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class CustomGRPOTrainer:
    def __init__(
        self,
        df_dict,
        policy_model_name: str,
        ref_model_name: str = None,
    ):
        # Store args
        self.df_dict = df_dict
        self.policy_model_name = policy_model_name
        self.ref_model_name = ref_model_name or policy_model_name
        self.sim_model_name = sim_model_name
        self.tox_model_name = tox_model_name
        self.categorical_weights = categorical_weights or {"toxicity": 1.0}
        self.sizes = dict(test=test_size, eval=eval_size, rl=rl_size)
        self.output_dir = output_dir
        # GRPO config
        self.grpo_config = GRPOConfig(
            group_size=group_size,
            clipping_eps=clipping_eps,
            lambda_kl=lambda_kl,
            entropy_penalty=entropy_penalty,
        )

        # Initialize components
        self._init_tokenizers_and_models()
        self._init_loss_fn()
        self._init_datasets()
        self._init_data_collator()
        self._init_grpo_trainer()

    def _init_tokenizers_and_models(self):
        # Policy & reference
        self.tokenizer = AutoTokenizer.from_pretrained(self.policy_model_name)
        self.policy_model = AutoModelForSeq2SeqLM.from_pretrained(self.policy_model_name)
        self.ref_model = AutoModelForSeq2SeqLM.from_pretrained(self.ref_model_name).eval()
        # Similarity & toxicity
        self.sim_tok = AutoTokenizer.from_pretrained(self.sim_model_name)
        self.sim_mod = AutoModel.from_pretrained(self.sim_model_name)
        self.tox_tok = AutoTokenizer.from_pretrained(self.tox_model_name)
        self.tox_mod = AutoModelForSequenceClassification.from_pretrained(self.tox_model_name)

    def _init_loss_fn(self):
        self.loss_fn = SimilarityToxicityLoss(
            sim_tokenizer=self.sim_tok,
            sim_model=self.sim_mod,
            tox_tokenizer=self.tox_tok,
            tox_model=self.tox_mod,
            categorical_weights=self.categorical_weights,
            w_sim=1.0,
            w_tox=1.0,
        )

    def _init_datasets(self):
        # Build datasets from df_dict
        dataset = Dataset.from_dict(self.df_dict)
        # Test split
        split1 = dataset.train_test_split(test_size=self.sizes['test'], seed=42)
        rest, self.test_dataset = split1['train'], split1['test']
        # Eval split
        eval_frac = self.sizes['eval'] / (1 - self.sizes['test'])
        split2 = rest.train_test_split(test_size=eval_frac, seed=42)
        rest, self.eval_dataset = split2['train'], split2['test']
        # RL split
        rl_frac = self.sizes['rl'] / (1 - self.sizes['test'] - self.sizes['eval'])
        split3 = rest.train_test_split(test_size=rl_frac, seed=42)
        self.sft_dataset, self.rl_dataset = split3['train'], split3['test']

    def _init_data_collator(self):
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.policy_model,
            padding=True,
            return_tensors="pt"
        )

    def _reward_fn(self, samples, prompts, model_inputs=None, **kwargs):
        # sim + toxicity as loss (lower => better)
        loss_vals = self.loss_fn(prompts, samples)  # tensor (B,)
        # KL penalty
        logprobs, ref_logprobs = trl_utils.logprobs_of_sequences(
            self.policy_model, self.tokenizer,
            samples, model_inputs,
            ref_model=self.ref_model
        )
        kl_vals = (logprobs - ref_logprobs).sum(dim=-1)  # tensor (B,)
        # reward: negative loss - KL
        rewards = -loss_vals.detach() - self.grpo_config.lambda_kl * kl_vals.detach()
        return rewards.tolist()

    def _init_grpo_trainer(self):
        self.trainer = GRPOTrainer(
            model=self.policy_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.rl_dataset,
            data_collator=self.data_collator,
            reward_fn=self._reward_fn,
            config=self.grpo_config,
        )

    def train(self, total_steps=None):
        # Run GRPO training
        self.trainer.train(total_steps=total_steps)
        # Save outputs
        os.makedirs(self.output_dir, exist_ok=True)
        self.policy_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Saved fine-tuned model to {self.output_dir}")