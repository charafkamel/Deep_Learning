import yaml
from huggingface_hub import login, HfApi
import wandb
import os

def load_tokens(yaml_file):
    """
    Loads tokens from a YAML file.
    
    Parameters:
        yaml_file (str): Path to the YAML file containing the tokens
    
    Returns:
        dict: Dictionary with loaded tokens
    """
    with open(yaml_file, 'r') as file:
        tokens = yaml.safe_load(file)
    return tokens

def huggingface_login(hf_token: str):
    """
    Logs into Hugging Face using the provided token.
    
    Parameters:
        hf_token (str): Hugging Face API token
    """
    try:
        login(token=hf_token)
        print("Logged into Hugging Face successfully!")
    except Exception as e:
        print(f"Error logging into Hugging Face: {e}")

def huggingface_list_models():
    """
    Lists models for the logged-in user on Hugging Face.

    Returns:
        list: List of models for the user
    """
    try:
        api = HfApi()
        user_models = api.list_models()
        print("User Models: ", user_models)
        return user_models
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def wandb_login(wb_token: str):
    """
    Logs into Weights & Biases using the provided API key.
    
    Parameters:
        wb_token (str): Weights & Biases API key
    """
    try:
        wandb.login(key=wb_token)
        print("Logged into WandB successfully!")
    except Exception as e:
        print(f"Error logging into WandB: {e}")

def user_login(logger="", path_to_src=""):
    
    token_filepath = os.path.join(path_to_src, "tokens.yaml")
    print(f"Token file path: {token_filepath}")
    # Load tokens from YAML file
    tokens = load_tokens(token_filepath)

    print("-------")
    # Use tokens to log in
    huggingface_login(tokens.get('huggingface_token'))
    wandb_login(tokens.get('wandb_token'))

    # List Hugging Face models (if needed)
    huggingface_list_models()
    
    if(logger):
        logger.info(f"Login done")


