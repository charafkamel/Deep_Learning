import os
import logging
import yaml
from grpo_trainer import CustomGRPOTrainer
from helpers import user_login
from logger import CustomLogger
from data_processing import process_dataset_and_split
print(os.getcwd())
from sft_trainer_base import SFTTrainerBase
from sft_trainer_count import CustomCountTrainer
from sft_trainer_generative_base import SFTGenerativeTrainerBase
from sft_trainer_generative_count import CustomGenerativeCountTrainer

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
    base = args.base
    rl = args.rl
    count = args.count
    base_generative = args.base_generative
    count_generative = args.count_generative
    #if not sft and not rl:
    #    logger.error("Both SFT and RL training are disabled. What to do? :D")

    process_dataset_and_split(logger, print_stats=True)
    logger.info("Dataset processing and splitting completed.")
    
    if base:
        logger.info("Starting SFT training...")
        custom_trainer_base = SFTTrainerBase(
            "google/t5-v1_1-base",
            config=config,
            logger=logger
        )
        custom_trainer_base.trainer.train(resume_from_checkpoint=False)
    if base_generative:
        logger.info("Starting SFT training...")
        custom_trainer_base_generative = SFTGenerativeTrainerBase(
            "Qwen/Qwen3-0.6B",
            config=config,
            logger=logger
        )
        
        custom_trainer_base_generative.trainer.train(resume_from_checkpoint=False)
    
    if rl:
        logger.info("Starting RL training...")
        custom_grpo_trainer = CustomGRPOTrainer(
            policy_model_name="Qwen/Qwen3-0.6B",
            config=config,
            logger=logger,
        )
        custom_grpo_trainer.trainer.train(resume_from_checkpoint=False)
        logger.error("RL training is not implemented yet.")
    
    if count:
        logger.info("Starting COUNT-based detoxification training...")
        count_trainer = CustomCountTrainer(
            model_name="google/t5-v1_1-base",
            tox_model_name="unitary/toxic-bert",
            config=config,
            logger=logger
        )
        count_trainer.trainer.train(resume_from_checkpoint=False)
    if count_generative:
        logger.info("Starting COUNT-based detoxification training...")
        count_trainer_generative = CustomGenerativeCountTrainer(
            "Qwen/Qwen3-0.6B",
            config=config,
            logger=logger
        )
        
        count_trainer_generative.trainer.train(resume_from_checkpoint=False)
        logger.info("COUNT-based detoxification training completed.")
    
    exit(-1)

if __name__ == "__main__":
    ### Give me an argument boolean that allows sft or not default to true
    ### Same for rl default to true
    ### Add a help message for each argument
    import argparse
    parser = argparse.ArgumentParser(description="Run SFT and RL training.")
    parser.add_argument("--base", action='store_true',
                        help="Enable SFT (if enabled, True; otherwise, False)")
    parser.add_argument("--base_generative", action='store_true',
                    help="Enable SFT (if enabled, True; otherwise, False)")
    parser.add_argument("--rl", action='store_true',
                        help="Enable RL (if enabled, True; otherwise, False)")
    parser.add_argument("--count_generative", action='store_true',
                        help="Enable COUNT-based detoxification training (if enabled, True; otherwise, False)")
    parser.add_argument("--count", action='store_true',
                        help="Enable COUNT-based detoxification training (if enabled, True; otherwise, False)")
    args = parser.parse_args()
    main(args)


## Script
