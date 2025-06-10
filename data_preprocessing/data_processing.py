from datasets import Dataset
import pandas as pd
import os

def process_dataset_and_split(logger=None, print_stats=True):
    """
    Process the dataset and split it into SFT, RL, eval, and test datasets.

    Parameters:
        logger (logging.Logger): Logger instance for logging messages.
        print_stats (bool): Whether to print dataset statistics.

    Returns:
        tuple: Contains SFT, RL, eval, and test datasets as Hugging Face Dataset objects.
    """

    # 1. create data directory if it does not exist
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    cleaned_dir = os.path.join(current_dir, 'cleaned_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 2. load the paradetox dataset and rename columns
    df = pd.read_csv(data_dir +'/paradetox.tsv', sep='\t')

    df.rename(columns={'neutral1': 'neutral'}, inplace=True)

    filtered_df = df[["toxic", "neutral"]]
    df_dict = filtered_df.to_dict(orient="records")

    dict_of_lists = {
        "toxic":   [e["toxic"]   for e in df_dict],
        "neutral": [e["neutral"] for e in df_dict],
    }
    dataset = Dataset.from_dict(dict_of_lists)

    # 3. Split first into train and test datasets
    split1 = dataset.train_test_split(test_size=0.05, seed=42)
    rest, test_dataset = split1["train"], split1["test"]

    # 4. Split the remaining 95% into 5% eval and 90% rest
    split2 = rest.train_test_split(test_size=0.05 / 0.95, seed=42)
    rest, eval_dataset = split2["train"], split2["test"]

    # 5. Split the remaining 90% into SFT (70%) and RL (30%) datasets
    split3 = rest.train_test_split(test_size=0.30 / 0.70, seed=42)
    sft_dataset, rl_dataset = split3["train"], split3["test"]

    if print_stats:
        logger.info("Dataset sizes after splitting:")
        logger.info(f"Total size: {len(dataset)}")
        logger.info(f"SFT size: {len(sft_dataset)}")
        logger.info(f"RL size: {len(rl_dataset)}")
        logger.info(f"Eval size: {len(eval_dataset)}")
        logger.info(f"Test size: {len(test_dataset)}")
    
    # 6. Save the datasets to disk distributed
    sft_dataset.save_to_disk(os.path.join(cleaned_dir, "sft_dataset"))
    rl_dataset.save_to_disk(os.path.join(cleaned_dir, "rl_dataset"))
    eval_dataset.save_to_disk(os.path.join(cleaned_dir, "eval_dataset"))
    test_dataset.save_to_disk(os.path.join(cleaned_dir, "test_dataset"))

    # 7. Return those datasets
    return sft_dataset, rl_dataset, eval_dataset, test_dataset

def load_dataset_from_disk(split_name):

    """
    Load a dataset from disk based on the split name.
    Parameters:
        split_name (str): The name of the dataset split to load (e.g., "sft_dataset", "rl_dataset", "eval_dataset", "test_dataset").
    Returns:
        Dataset: The loaded Hugging Face Dataset object.
    """

    current_dir = os.getcwd()
    cleaned_dir = os.path.join(current_dir, 'cleaned_data')
    dataset_path = os.path.join(cleaned_dir, split_name)
    return Dataset.load_from_disk(dataset_path)
