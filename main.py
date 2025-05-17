import os
import logging
import yaml
from helpers import user_login
from logger import CustomLogger
from data_processing import process_dataset_and_split
from sft_trainer import CustomTrainer

def load_config(config_path="main_config.yml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main(args):
    
    # Load configuration from YAML
    config = load_config()

    # Setup logging
    logger = CustomLogger(__name__, level=logging.DEBUG)

    # User login
    user_login(logger)

    ## parse sft and rl from args
    sft = args.sft
    rl = args.rl
    if not sft and not rl:
        logger.error("Both SFT and RL training are disabled. What to do? :D")

    process_dataset_and_split(logger, print_stats=True)
    logger.info("Dataset processing and splitting completed.")
    
    if sft:
        logger.info("Starting SFT training...")
        custom_trainer = CustomTrainer(
            "google/t5-v1_1-base",
            config=config,
            logger=logger
        )

        #device = custom_trainer.model.device if hasattr(custom_trainer.model, "device") else custom_trainer.trainer.args.device
        #batch = next(iter(custom_trainer.trainer.get_train_dataloader()))
        #batch = {k: v.to(device) for k, v in batch.items()}
        #outputs = custom_trainer.model(**batch)
        #print("⚠ raw model outputs.loss:", outputs.loss)
        #print("⚠ batch labels stats:", 
        #      batch["labels"].unique()[:10], 
        #      "count of -100:", (batch["labels"] == -100).sum().item(), 
        #      "total tokens:", batch["labels"].numel())
        
        custom_trainer.trainer.train(resume_from_checkpoint=False)
    
    if rl:
        logger.info("Starting RL training...")
        logger.error("RL training is not implemented yet.")

    exit(-1)

if __name__ == "__main__":
    ### Give me an argument boolean that allows sft or not default to true
    ### Same for rl default to true
    ### Add a help message for each argument
    import argparse
    parser = argparse.ArgumentParser(description="Run SFT and RL training.")
    parser.add_argument("--sft", type=bool, default=True, help="Run SFT training")
    parser.add_argument("--rl", type=bool, default=True, help="Run RL training")
    args = parser.parse_args()
    main(args)


## Script
