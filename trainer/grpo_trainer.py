import os
from utils.custom_loss import SimilarityToxicityReward
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk

# ############################################# #
# CustomGRPOTrainer with custom reward function #
# ############################################# #
class CustomGRPOTrainer:
    def __init__(self, policy_model_name, config, logger=None):
        self.model_name = policy_model_name
        self.config = config
        self.logger = logger
        self.repo_id = self.config.get("hf_username", "kamelcharaf")
        self.model_name_for_hub = policy_model_name.split("/")[-1]
        self.new_model_name = f"{self.repo_id}/GRPO-{self.model_name_for_hub}"
        self.output_dir = config.get("output_dir", "./grpo_results")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grpo_config = GRPOConfig(**config["grpo_params"], hub_model_id=self.new_model_name, run_name=self.new_model_name)
        
        ## Initialize components and log the process
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
        """
        Initialize the tokenizer and models for the policy and reference models.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        ## We use the same model for both policy and reference
        self.policy_model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_name).eval().to(self.device)

    def _init_datasets(self):
        """
        Load and preprocess the datasets for training and evaluation.
        """

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
        """
        Initialize the reward model for toxicity and similarity.
        """
        self.reward_model = SimilarityToxicityReward(w_sim=1.0, w_tox=1.0, device=self.device)


    def _reward_fn(self, prompts, completions, **kwargs):
        """
        Custom reward function that computes the reward based on our custom reward model and KL divergence penalty.
        Args:
            prompts (list): List of input prompts.
            completions (list): List of model-generated completions.
        Returns:
            list: List of computed rewards.
        """

        def _get_log_probs(model, tokenizer, texts):
            """
            Private helper function to compute log probabilities of tokens in the texts.
            Args:
                model: The model to compute log probabilities.
                tokenizer: The tokenizer to encode the texts.
                texts (list): List of input texts.
            Returns:
                torch.Tensor: Log probabilities of the tokens in the texts.
            """
            ## First we need to tokenize the texts with padding and truncation
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

            ## Make sure inputs are on the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            ## Forward pass to get logits
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            ## Compute log probabilities by shifting the logits and labels
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = inputs['input_ids'][:, 1:].contiguous()

            ## Gather the log probabilities using softmax
            log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
            log_probs_of_tokens = torch.gather(log_probs, -1, shifted_labels.unsqueeze(-1)).squeeze(-1)

            ## Attention mask is used to ignore padding tokens in the log probabilities
            attention_mask_shifted = inputs['attention_mask'][:, 1:]

            ## Sum the log probabilities of the tokens, ignoring padding tokens
            summed_log_probs = (log_probs_of_tokens * attention_mask_shifted).sum(dim=-1)
            return summed_log_probs

        ## Get task reward using the custom reward model    
        with torch.no_grad():
            task_reward = self.reward_model(prompts, completions).detach()
        
        ## Calculate the log probabilities for the policy and reference models
        policy_logprobs_sum = _get_log_probs(self.policy_model, self.tokenizer, completions)
        ref_logprobs_sum = _get_log_probs(self.ref_model, self.tokenizer, completions)

        ## Compute the KL divergence penalty and clamp it between 0 and 2 since each reward is in [0, 1]
        kl_vals = (policy_logprobs_sum - ref_logprobs_sum).detach()
        kl_pen = kl_vals.clamp(min=0.0, max=2.0)

        _lambda = 0.1 ## Hyperparameter for scaling the KL penalty

        ## Final reward is the task reward (our reward) minus the KL penalty scaled by _lambda
        final_reward = task_reward - _lambda * kl_pen

        ## Some logging for debugging purposes
        self.logger.debug(f"Task reward: {task_reward.tolist()}")
        self.logger.debug(f"Final reward: {final_reward.tolist()}")

        ## Return the final reward as a list
        return final_reward.tolist()

    def _init_grpo_trainer(self):
        """
        Initialize the GRPO trainer with the policy model, datasets, and reward function.
        """

        self.trainer = GRPOTrainer(
            model=self.policy_model,
            args=self.grpo_config,
            train_dataset=self.rl_dataset,
            eval_dataset=self.eval_dataset,
            reward_funcs=self._reward_fn
        )

    def train(self, resume_from_checkpoint=False):
        """
        Train the model using the GRPO trainer.
        Args:
            resume_from_checkpoint (bool): Whether to resume training from a checkpoint.
        """

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.output_dir = self.trainer.args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.policy_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.trainer.push_to_hub()
        print(f"Saved fine-tuned model to {self.output_dir}")