import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
os.environ["BITSANDBYTES_CUDA_VERSION"] = "none"

# ################################## #
# SFTTrainerBase for T5-based models #
# ################################## #
class SFTTrainerBase:
    def __init__(self, model_name="google/t5-v1_1-base", config=None, logger=None):
        self.config = config
        self.logger = logger
        self.repo_id = self.config.get("hf_username", "kamelcharaf")
        self.model_name = model_name

        ## Initialize components
        self.load_model_and_tokenizer(model_name)
        self.preprocess_and_split_dataset()
        self.setup_training_args_and_trainer()



    # 1) load model and tokenizer
    def load_model_and_tokenizer(self, model_name):
        """Load the model and tokenizer."""
        if self.logger:
            self.logger.info("Loading model and tokenizer")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)


    # 2) format data
    def preprocess_and_split_dataset(self):
        """Load and preprocess the dataset."""
        if self.logger:
            self.logger.info("Loading dataset from disk and preprocessing")
        # load the dataset from disk
        self.config["dataset_path"] = os.path.join(os.getcwd(), "cleaned_data", "sft_dataset")
        dataset = load_from_disk(self.config["dataset_path"])

        ## Mapper to get the format as detoxify: {toxic} and the target as neutral
        def _preprocess(example):
            input_text = f"detoxify: {example['toxic']}"
            target_text = example["neutral"]
            input_enc = self.tokenizer(input_text, truncation=True, max_length=64)
            target_enc = self.tokenizer(target_text, truncation=True,max_length=64)
            input_enc["labels"] = target_enc["input_ids"]
            return input_enc

        tokenized_dataset = dataset.map(_preprocess, remove_columns=dataset.column_names)
        split = tokenized_dataset.train_test_split(test_size=self.config.get("eval_split", 0.1), seed=self.config.get("seed", 42))
        self.train_dataset, self.eval_dataset = split["train"], split["test"]

    # 3) Setup training arguments
    def setup_training_args_and_trainer(self):
        if self.logger:
            self.logger.info("Setting up training arguments")

        ## We need to split on '/' to get the model name for the hub (if not, it will be treated as a path)
        self.model_name_for_hub = self.model_name.split("/")[-1]

        ## Create a new model name for the hub
        self.new_model_name = f"{self.repo_id}/sft-base_loss-{self.model_name_for_hub}-mle{0}-ul{0}-tox{0}-e{self.config['sft_params_base_generative']['num_train_epochs']}"
        
        ## Define the training arguments
        self.training_args = Seq2SeqTrainingArguments(**self.config["sft_params_base_enc_dec"], hub_model_id=self.new_model_name,
            run_name=  self.new_model_name)
        
        self.trainer = Seq2SeqTrainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.train_dataset,
            eval_dataset = self.eval_dataset,
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
    )

    ## 4) Train the model
    def train(self, resume_from_checkpoint=False):
        if self.logger:
            self.logger.info("Starting training")

        self.trainer.train(resume_from_checkpoint)

        ## Save the model and tokenizer
        out_dir = self.trainer.args.output_dir

        ## save model + tokenizer into output_dir/e{epoch}
        self.trainer.save_model(out_dir)
        self.tokenizer.save_pretrained(out_dir)

        self.trainer.push_to_hub()

        if self.logger:
            self.logger.info(f"Model & tokenizer saved to {out_dir}")
        return out_dir
