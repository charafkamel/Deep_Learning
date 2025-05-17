import os
import logging
import yaml
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from helpers import user_login
from logger import CustomLogger
from data_processing import load_dataset_from_disk
from transformers import DataCollatorForSeq2Seq


class CustomTrainer:
    def __init__(self, model_name="google/t5-v1_1-base", config=None, logger=None):
        self.config = config
        self.logger = logger
        self.load_model_and_tokenizer(model_name)
        self.preprocess_and_split_dataset()
        self.setup_training_args_and_trainer()



    # 1) load model and tokenizer
    def load_model_and_tokenizer(self, model_name):
        """Load the model and tokenizer."""
        if self.logger:
            self.logger.info("Loading model and tokenizer")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model    = T5ForConditionalGeneration.from_pretrained(model_name)


    # 2) format data
    
    def preprocess_and_split_dataset(self):
        """Load and preprocess the dataset."""
        if self.logger:
            self.logger.info("Loading dataset from disk and preprocessing")
        # load the dataset from disk
        self.config["dataset_path"] = os.path.join(os.getcwd(), "cleaned_data", "sft_dataset")
        dataset = load_from_disk(self.config["dataset_path"])

        ## Mapper
        def _preprocess(example):
            input_text = f"detoxify: {example['toxic']}"
            target_text = example["neutral"]
            input_enc = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=64)
            target_enc = self.tokenizer(target_text, truncation=True, padding="max_length", max_length=64)
            input_enc["labels"] = target_enc["input_ids"]
            return input_enc

        tokenized_dataset = dataset.map(_preprocess, remove_columns=dataset.column_names)
        split = tokenized_dataset.train_test_split(test_size=0.1)
        self.train_dataset, self.eval_dataset = split["train"], split["test"]

    # 3) Setup training arguments
    def setup_training_args_and_trainer(self):
        if self.logger:
            self.logger.info("Setting up training arguments")

        self.training_args = Seq2SeqTrainingArguments(**self.config["sft_params"])
        import os
        os.environ["BITSANDBYTES_CUDA_VERSION"] = "none"
        self.trainer = Seq2SeqTrainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.train_dataset,
            eval_dataset = self.eval_dataset,
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            #tokenizer = self.tokenizer,
    )

    ## 4) Train the model
    def train(self, resume_from_checkpoint=False):
        if self.logger:
            self.logger.info("Starting training")

        # run the training loop
        self.trainer.train(resume_from_checkpoint)

        final_epoch = int(self.trainer.state.epoch or 0)
        out_dir = os.path.join(self.trainer.args.output_dir, f"e{final_epoch}")
        os.makedirs(out_dir, exist_ok=True)

        # save model + tokenizer into output_dir/e{epoch}
        self.trainer.save_model(out_dir)
        self.tokenizer.save_pretrained(out_dir)

        if self.logger:
            self.logger.info(f"Model & tokenizer saved to {out_dir}")
        return out_dir
