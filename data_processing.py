from datasets import Dataset
import pandas as pd
import os
import logging
import yaml
from helpers import user_login
from logger import CustomLogger

def process_dataset_and_split(logger=None, print_stats=True):
## Read data/paradox.tsv file as pandas dataframe
# create data directory if it does not exist
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    cleaned_dir = os.path.join(current_dir, 'cleaned_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    df = pd.read_csv(data_dir +'/paradetox.tsv', sep='\t')

    ## rename neutral1 column to neutral
    df.rename(columns={'neutral1': 'neutral'}, inplace=True)

    filtered_df = df[["toxic", "neutral"]]
    df_dict = filtered_df.to_dict(orient="records")

    dict_of_lists = {
        "toxic":   [e["toxic"]   for e in df_dict],
        "neutral": [e["neutral"] for e in df_dict],
    }
    dataset = Dataset.from_dict(dict_of_lists)

    # 1) 5% for test
    split1 = dataset.train_test_split(test_size=0.05, seed=42)
    rest, test_dataset = split1["train"], split1["test"]

    # 2) carve off 5% for eval out of the remaining 95%
    #    → proportion = 0.05 / 0.95 ≈ 0.0526316
    split2 = rest.train_test_split(test_size=0.05 / 0.95, seed=42)
    rest, eval_dataset = split2["train"], split2["test"]

    # 3) split the remaining 90% into 10% RL vs. 80% SFT
    #    → rl fraction = 0.10 / 0.90 ≈ 0.1111111
    split3 = rest.train_test_split(test_size=0.30 / 0.70, seed=42)
    sft_dataset, rl_dataset = split3["train"], split3["test"]

    if print_stats:
        logger.info("Dataset sizes after splitting:")
        logger.info(f"Total size: {len(dataset)}")
        logger.info(f"SFT size: {len(sft_dataset)}")
        logger.info(f"RL size: {len(rl_dataset)}")
        logger.info(f"Eval size: {len(eval_dataset)}")
        logger.info(f"Test size: {len(test_dataset)}")
    
    #4) Save the datasets to disk
    sft_dataset.save_to_disk(os.path.join(cleaned_dir, "sft_dataset"))
    rl_dataset.save_to_disk(os.path.join(cleaned_dir, "rl_dataset"))
    eval_dataset.save_to_disk(os.path.join(cleaned_dir, "eval_dataset"))
    test_dataset.save_to_disk(os.path.join(cleaned_dir, "test_dataset"))
    return sft_dataset, rl_dataset, eval_dataset, test_dataset

def load_dataset_from_disk(split_name):
    """Load dataset from disk."""
    current_dir = os.getcwd()
    cleaned_dir = os.path.join(current_dir, 'cleaned_data')
    dataset_path = os.path.join(cleaned_dir, split_name)
    return Dataset.load_from_disk(dataset_path)
