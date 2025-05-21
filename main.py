import os
import logging
import yaml
from grpo_trainer import CustomSeq2SeqRLTrainer
from helpers import user_login
from logger import CustomLogger
from data_processing import process_dataset_and_split
from sft_trainer import CustomTrainer
import torch
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
        exit(-1)

    process_dataset_and_split(logger, print_stats=True)
    logger.info("Dataset processing and splitting completed.")
    
    if sft:
        logger.info("Starting SFT training...")
        custom_trainer = CustomTrainer(
            "google/t5-v1_1-base",
            config=config,
            logger=logger
        )
        custom_trainer.trainer.train(resume_from_checkpoint=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
        #device = custom_trainer.model.device if hasattr(custom_trainer.model, "device") else custom_trainer.trainer.args.device
        #batch = next(iter(custom_trainer.trainer.get_train_dataloader()))
        #batch = {k: v.to(device) for k, v in batch.items()}
        #outputs = custom_trainer.model(**batch)
        #print("⚠ raw model outputs.loss:", outputs.loss)
        #print("⚠ batch labels stats:", 
        #      batch["labels"].unique()[:10], 
        #      "count of -100:", (batch["labels"] == -100).sum().item(), 
        #      "total tokens:", batch["labels"].numel())
        
    
    
    if rl:
        logger.info("Starting RL training...")

        # 3) Instantiate your trainer
        logger.info("Instantiating GRPO trainer...")
        trainer = CustomSeq2SeqRLTrainer( 
            config=config,
            device=device,
            logger=logger,
            )

        # 4) Kick off training
        #    - total_steps: how many RL updates to perform  
        #    - resume_from_checkpoint: set True if you want to pick up from an existing RL checkpoint
        logger.info("Starting GRPO training...")
        trainer.train(resume_from_checkpoint=False)

    exit(-1)

if __name__ == "__main__":
    ### Give me an argument boolean that allows sft or not default to true
    ### Same for rl default to true
    ### Add a help message for each argument
    import argparse
    parser = argparse.ArgumentParser(description="Run SFT and RL training.")
    parser.add_argument("--sft", action='store_true',
                    help="Enable Sft (if enabled, True; otherwise, False)")
    parser.add_argument("--rl",action='store_true',
                    help="Enable RK (if enabled, True; otherwise, False)")
    args = parser.parse_args()
    main(args)


## Script
