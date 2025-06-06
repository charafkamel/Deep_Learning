import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from utils.helpers import user_login
from utils.logger import CustomLogger
from data_preprocessing.data_processing import load_dataset_from_disk

# ################################################################ #
# SFTGenerativeTrainerBase for training generative models with SFT #
# ################################################################ #
class SFTGenerativeTrainerBase:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", config=None, logger=None):
        self.config = config
        self.logger = logger
        self.repo_id = self.config.get("hf_username", "kamelcharaf")
        self.model_name = model_name
        

        ## Initialize components
        self.load_model_and_tokenizer(model_name)
        self.preprocess_and_split_dataset()
        self.setup_training_args_and_trainer()

    def load_model_and_tokenizer(self, model_name):
        """Load the model and tokenizer."""
        if self.logger:
            self.logger.info("Loading model and tokenizer")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        ## Ensure proper padding and EOS handling
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def preprocess_and_split_dataset(self):
        """Load and preprocess the dataset."""
        if self.logger:
            self.logger.info("Loading and preprocessing dataset")

        self.config["dataset_path"] = os.path.join(os.getcwd(), "cleaned_data", "sft_dataset")
        dataset = load_from_disk(self.config["dataset_path"])


        def _preprocess(example):
            """
            Private helper function to preprocess the dataset.
            Converts the toxic text to a prompt and the neutral text to the target.
            """

            prompt = f"detoxify: {example['toxic']}\nNeutral:"
            full_text = f"{prompt} {example['neutral']}"
            return {"text": full_text}

        ## Tokenize the dataset and split for training and evaluation
        tokenized_dataset = dataset.map(_preprocess, remove_columns=dataset.column_names)
        split = tokenized_dataset.train_test_split(test_size=self.config.get("eval_split", 0.1), seed=self.config.get("seed", 42))
        self.train_dataset, self.eval_dataset = split["train"], split["test"]

    
    def setup_training_args_and_trainer(self):
        """
        Setup training arguments and initialize the SFTTrainer.
        """

        if self.logger:
            self.logger.info("Setting up training arguments and SFTTrainer")

        ## We split on '/' to get the model name for the hub (if not, it will be treated as a path)
        self.model_name_for_hub = self.model_name.split("/")[-1]

        ## Define the new model name for the hub
        self.new_model_name = f"{self.repo_id}/sft-base_loss-{self.model_name_for_hub}-mle{0}-ul{0}-tox{0}-e{self.config['sft_params_base_generative']['num_train_epochs']}"
        
        ## Define the training arguments
        self.training_args = SFTConfig(**self.config["sft_params_base_generative"], hub_model_id=self.new_model_name,
            run_name=  self.new_model_name)

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

    def train(self, resume_from_checkpoint=False):
        """
        Train the model using the SFTTrainer.
        Args:
            resume_from_checkpoint (bool): Whether to resume training from a checkpoint.
        """

        if self.logger:
            self.logger.info("Starting training")

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        ## Save the model and tokenizer after training
        final_epoch = int(self.trainer.state.epoch or 0)
        out_dir = os.path.join(self.trainer.args.output_dir, f"e{final_epoch}")
        os.makedirs(out_dir, exist_ok=True)

        self.trainer.save_model(out_dir)
        self.tokenizer.save_pretrained(out_dir)

        if self.logger:
            self.logger.info(f"Model & tokenizer saved to {out_dir}")
        self.trainer.push_to_hub()
        return out_dir