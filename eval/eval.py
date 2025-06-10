import logging
import torch
import re
import pandas as pd
import torch.nn.functional as F
from huggingface_hub import list_models
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModelForSeq2SeqLM,
)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from data_preprocessing.data_processing import load_dataset_from_disk
from utils.helpers import load_config, user_login
from utils.logger import CustomLogger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def result_table():
    # Load CSV
    df = pd.read_csv("eval/results/detoxification_evaluation_merged.csv")

    # Group and aggregate
    agg = df.groupby("model_id").agg(
        similarity_mean=("similarity_original_generated", "mean"),
        similarity_std=("similarity_original_generated", "std"),
        toxicity_mean=("toxicity_generated", "mean"),
        toxicity_std=("toxicity_generated", "std"),
        n=("similarity_original_generated", "count")
    ).reset_index()

    # Compute combined score
    agg["combined_score"] = 0.5 * agg["similarity_mean"] + 0.5 * (1 - agg["toxicity_mean"])

    # Sort by combined score
    agg = agg.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # Add rank
    agg["rank"] = agg.index + 1

    # Format strings
    agg["Similarity (mean ± std)"] = agg.apply(
        lambda row: f"{row['similarity_mean']:.2f} ± {row['similarity_std']:.2f}", axis=1)
    agg["Toxicity (mean ± std)"] = agg.apply(
        lambda row: f"{row['toxicity_mean']:.2f} ± {row['toxicity_std']:.2f}", axis=1)
    agg["Combined Score"] = agg["combined_score"].round(3)
    agg["Rank"] = agg["rank"]

    # Final display table
    summary_df = agg[[
        "model_id",
        "Similarity (mean ± std)",
        "Toxicity (mean ± std)",
        "Combined Score",
    ]].rename(columns={"model_id": "Model"})

    # Ensure output directory exists
    os.makedirs("../visualizations", exist_ok=True)

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, len(summary_df) * 0.5))
    ax.axis("off")  # Hide axes

    # Create table in the figure
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Adjust for spacing

    # Save to file
    plt.savefig("../visualizations/table.png", bbox_inches="tight", dpi=300)
    plt.close()

def visualizer_histogram():
        
    # Load the full dataset
    df = pd.read_csv("eval/results/detoxification_evaluation_merged.csv")

    # Count number of samples per model
    counts = df.groupby("model_id").size().reset_index(name="n")

    # Group by model_id and compute mean and std
    agg_df = df.groupby("model_id").agg({
        "toxicity_generated": ["mean", "std"],
        "similarity_original_generated": ["mean", "std"]
    }).reset_index()

    # Flatten columns
    agg_df.columns = [
        "model_id",
        "toxicity_generated", "std_toxicity_generated",
        "similarity_original_generated", "std_similarity_original_generated"
    ]

    # Merge with sample counts
    df = agg_df.merge(counts, on="model_id")

    # Drop models with invalid names
    df = df[~df['model_id'].str.startswith('_')]

    # Compute combined score
    df['combined_score'] = 0.5 * df['similarity_original_generated'] + 0.5 * (1 - df['toxicity_generated'])

    # Compute standard error of the mean for each
    df['se_sim'] = df['std_similarity_original_generated'] / np.sqrt(df['n'])
    df['se_tox'] = df['std_toxicity_generated'] / np.sqrt(df['n'])

    # Compute standard error of the combined score (assuming independence)
    df['se_combined'] = 0.5 * df['se_sim'] + 0.5 * df['se_tox']

    # Sort by combined score
    df = df.sort_values(by='combined_score', ascending=False).reset_index(drop=True)

    # Extract values
    models = df["model_id"]
    similarity = df["similarity_original_generated"]
    toxicity = df["toxicity_generated"]
    combined_score = df["combined_score"]
    std_sim = df["se_sim"]
    std_tox = df["se_tox"]
    std_combined = df["se_combined"]

    # Set up positions
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12, 8), dpi=1000)

    # Plot main combined score with error bars
    ax.barh(x, combined_score, height=0.6, xerr=std_combined, color='mediumseagreen', label='Combined Score', capsize=4)

    # Add similarity and toxicity with scaled SE error bars
    ax.barh(x - 0.2, similarity, height=0.2, color='steelblue', label='Similarity', capsize=3)
    ax.barh(x + 0.2, toxicity, height=0.2, color='darkorange', label='Toxicity', capsize=3)

    # Plot aesthetics
    ax.set_xlabel('Score')
    ax.set_title('Model Performance: Combined Score vs. Similarity & Toxicity')
    ax.set_yticks(x)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    output_path = "../visualizations/Histogram.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)


def is_seq2seq_model(model_id, logger=None):
    """
    Checks if a model is a sequence-to-sequence model based on its configuration.
    Parameters:
        model_id (str): The identifier of the model.
    Returns:
        bool: If the given model_id is a seq2seq model
    """
    try:
        config = AutoConfig.from_pretrained(model_id)
        return config.architectures and any(
            "Seq2Seq" in a or "ConditionalGeneration" in a for a in config.architectures
        )
    except Exception as e:
        if logger:
            logger.warning(f"Could not load config for {model_id}. Assuming not seq2seq. Error: {e}")
        else:
            print(f"Warning: Could not load config for {model_id}. Assuming not seq2seq. Error: {e}")
        return False


def is_base_model(model_id):
    """
    Checks if the model is one of the predefined base models.
    Parameters:
        model_id (str): The identifier of the model.
    Returns:
        bool: If the model given is a base model or not.
    """
    return model_id in {"google/flan-t5-base", "Qwen/Qwen2.5-0.5B-Instruct"}


def build_prompt(model_id, toxic_sentence):
    """
    Constructs the appropriate prompt for the given model and toxic sentence.
    Parameters:
        model_id (str): The identifier of the model.
        toxic_sentence (str): The toxic sentence to detoxify.
    Returns:
        str: The constructed prompt for the model.
    """

    if model_id == "s-nlp/gpt2-base-gedi-detoxification":
        return (
            f"Change the toxic sentence into a neutral one.\n"
            f"Keep it as close as possible to the original. Only remove or replace toxic parts.\n"
            f"Just give neutral sentence alone.\n"
            f"Remember: You are allowed to generate only 1 sentence, that is neutral.\n"
            f"Toxic: {toxic_sentence}\nNeutral:"
        )
    elif model_id == "s-nlp/t5-paranmt-detox" or model_id == "textdetox/mbart-detox-baseline":
        return toxic_sentence
    elif is_base_model(model_id):
        if not is_seq2seq_model(model_id):
            return (
                f"You are given a toxic sentence. Your task is to rewrite it as a neutral sentence.\n"
                f"Respond with ONLY the neutral sentence, nothing else.\n"
                f"Toxic: {toxic_sentence}\nNeutral:" # Changed #### to just Neutral:
            )
        else:
            return f"detoxify: {toxic_sentence}"
    else:
        return f"detoxify: {toxic_sentence}" if is_seq2seq_model(model_id) else f"detoxify: {toxic_sentence}\nNeutral:"


def generate_outputs(model_id, eval_data, max_length=64):
    """
    Generates detoxified outputs for the given model and evaluation data.
    Parameters:
        model_id (str): The identifier of the model to use for generation.
        eval_data (list): The evaluation data containing toxic sentences.
        max_length (int): The maximum length of the generated output.
    Returns:
        list: A list of dictionaries containing original toxic sentences, reference neutral sentences, and generated neutral sentences.
    """
    is_seq2seq = is_seq2seq_model(model_id)

    if is_seq2seq:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        tokenizer.padding_side = "right"

        if model_id == "textdetox/mbart-detox-baseline":
            if hasattr(tokenizer, "lang_code_to_id") and "en_XX" in tokenizer.lang_code_to_id:
                forced_bos_token_id_val = tokenizer.lang_code_to_id["en_XX"]
            else:
                print(f"Warning: '{model_id}' tokenizer does not have 'lang_code_to_id' or 'en_XX'. Generation might be incorrect without forced_bos_token_id.")
                forced_bos_token_id_val = None

            generate_fn = lambda m, t: m.generate(
                **t,
                max_length=max_length,
                num_beams=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                forced_bos_token_id=forced_bos_token_id_val
            )
        else:
            generate_fn = lambda m, t: m.generate(
                **t,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        generate_fn = lambda m, t: m.generate(**t, max_new_tokens=max_length, pad_token_id=tokenizer.pad_token_id)

    model.eval()
    results = []

    for ex in eval_data:
        orig = ex["toxic"]
        ref = ex["neutral"]
        prompt = build_prompt(model_id, orig)

        if is_seq2seq:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
        else:
            enc = tokenizer(prompt, return_tensors="pt", truncation=False, padding=True).to(device)


        with torch.no_grad():
            out_ids = generate_fn(model, enc)

        gen = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        if not is_seq2seq:
            if model_id == "s-nlp/gpt2-base-gedi-detoxification":
                if "Neutral:" in gen:
                    gen = gen.split("Neutral:", 1)[-1].strip()
            elif is_base_model(model_id) and "Qwen" in model_id:
                gen_after_prompt = gen[len(prompt):].strip()

                extracted_text = ""
                if "Neutral:" in gen_after_prompt:
                    temp_gen = gen_after_prompt.split("Neutral:", 1)[-1].strip()
                    first_line = temp_gen.split('\n', 1)[0].strip()
                    if '.' in first_line or '!' in first_line or '?' in first_line:
                        
                        match = re.match(r"([^.!?]*[.!?])", first_line)
                        extracted_text = match.group(1).strip() if match else first_line # Take first sentence
                    else:
                        extracted_text = first_line
                elif "####" in gen_after_prompt: # Fallback if they somehow included #### again
                    temp_gen = gen_after_prompt.split("####", 1)[-1].strip()
                    first_line = temp_gen.split('\n', 1)[0].strip()
                    if '.' in first_line or '!' in first_line or '?' in first_line:
                        import re
                        match = re.match(r"([^.!?]*[.!?])", first_line)
                        extracted_text = match.group(1).strip() if match else first_line
                    else:
                        extracted_text = first_line
                else:
                    first_line = gen_after_prompt.split('\n', 1)[0].strip()
                    if '.' in first_line or '!' in first_line or '?' in first_line:
                        import re
                        match = re.match(r"([^.!?]*[.!?])", first_line)
                        extracted_text = match.group(1).strip() if match else first_line
                    else:
                        extracted_text = first_line

                gen = extracted_text.replace("\"", "").strip() # Remove any leftover quotes and extra whitespace
            else:
                gen = gen[len(prompt):].strip() # Default

        results.append({
            "original_toxic": orig,
            "reference_neutral": ref,
            "generated_neutral": gen
        })

    return results


def score_toxicity_and_similarity(results, logger):
    """
    Scores the toxicity and semantic similarity of the generated sentences.
    """
    tox_tok = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    tox_mod = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device).eval()
    label2id = tox_mod.config.label2id
    
    if "toxicity" in label2id:
        tox_label = label2id["toxicity"]
    elif "LABEL_0" in label2id:
        tox_label = label2id["LABEL_0"]
        if logger:
            logger.warning("Warning: 'toxicity' label not found for toxic-bert, defaulting to 'LABEL_0'.")
        else:
            print("Warning: 'toxicity' label not found for toxic-bert, defaulting to 'LABEL_0'.")
    else:
        tox_label = list(label2id.values())[0]
        print(f"Warning: 'toxicity' label not found for toxic-bert, defaulting to first label: {list(label2id.keys())[0]}. Please verify.")

    sim_tok = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    sim_mod = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device).eval()

    origs = [r["original_toxic"] for r in results]
    refs = [r["reference_neutral"] for r in results]
    gens = [r["generated_neutral"] for r in results]

    with torch.no_grad():
        tox_inputs = tox_tok(origs + refs + gens, return_tensors="pt", truncation=True, padding=True).to(device)
        tox_logits = tox_mod(**tox_inputs).logits
        tox_probs = torch.sigmoid(tox_logits)

        sim_inputs = sim_tok(origs + refs + gens, return_tensors="pt", truncation=True, padding=True).to(device)
        emb_all = sim_mod(**sim_inputs).pooler_output

    n = len(results)
    tox_orig, tox_ref, tox_gen = tox_probs[:n, tox_label], tox_probs[n:2 * n, tox_label], tox_probs[2 * n:, tox_label]
    sim_orig, sim_ref, sim_gen = emb_all[:n], emb_all[n:2 * n], emb_all[2 * n:]

    for i, r in enumerate(results):
        r["toxicity_orig"] = float(tox_orig[i].item())
        r["toxicity_ref"] = float(tox_ref[i].item())
        r["toxicity_gen"] = float(tox_gen[i].item())
        r["similarity_orig_ref"] = float(F.cosine_similarity(sim_orig[i], sim_ref[i], dim=0).item())
        r["similarity_orig_gen"] = float(F.cosine_similarity(sim_orig[i], sim_gen[i], dim=0).item())
    return results


def evaluate_model(model_id, eval_data, logger):
    """
    Evaluates a single model for detoxification and similarity.
    """
    results = generate_outputs(model_id, eval_data)
    results = score_toxicity_and_similarity(results, logger)
    print(f"Evaluation completed for model: {model_id}")
    df = pd.DataFrame(results)[[
        "original_toxic",
        "toxicity_orig",
        "reference_neutral",
        "toxicity_ref",
        "similarity_orig_ref",
        "generated_neutral",
        "toxicity_gen",
        "similarity_orig_gen"
    ]]
    df.rename(columns={
        "toxicity_orig": "toxicity_original",
        "toxicity_ref": "toxicity_reference",
        "toxicity_gen": "toxicity_generated",
        "similarity_orig_ref": "similarity_original_reference",
        "similarity_orig_gen": "similarity_original_generated"
    }, inplace=True)
    df.insert(0, "model_id", model_id)
    return df


def main():
    """
    Main function to search, evaluate, and save results for detoxification models.
    """
    config = load_config()

    # Setup logging
    logger = CustomLogger(__name__, level=logging.DEBUG)

    # User login
    user_login(logger)
    
    logger.info("Starting detoxification evaluation...")
    all_models = list_models(author=config.get("hf_username", "kamelcharaf"))
    sft_models = [m.modelId for m in all_models]

    if not sft_models:
        logger.error("No matching models found under TarhanE/sft-*.")

    logger.info("Loading evaluation dataset...")
    eval_ds = load_dataset_from_disk("test_dataset")
    eval_data = eval_ds.shuffle(seed=42) # Limit for fast debug

    base_models = [
        "google/flan-t5-base",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "textdetox/mbart-detox-baseline",
        "s-nlp/t5-paranmt-detox",
    ]
    all_models_to_eval = sft_models + base_models

    all_dfs = []
    for model_id in tqdm(all_models_to_eval, desc="Evaluating models"):
        try:
            logger.info(f"Evaluating model: {model_id}")
            df = evaluate_model(model_id, eval_data, logger)
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to evaluate {model_id}: {e}")
    

    ## create a dummy dataframe to save if no models were evaluated
    if not all_dfs:
        logger.warning("No models were evaluated successfully. Creating an empty DataFrame.")
        all_dfs = [pd.DataFrame(columns=[
            "model_id", "original_toxic", "toxicity_original", "reference_neutral",
            "toxicity_reference", "similarity_original_reference", "generated_neutral",
            "toxicity_generated", "similarity_original_generated"
        ])]
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv("eval/results/detoxification_evaluation_merged.csv", index=False)
        logger.info("Saved all results to: eval/results/detoxification_evaluation_merged.csv")
    else:
        logger.warning("No evaluations completed.")

    
    ## Start visualizaton
    logger.info("Generating result table and histogram visualizations...")
    result_table()
    visualizer_histogram()
    logger.info("Visualizations saved to ../visualizations/")
    logger.info("Detoxification evaluation completed successfully.")


if __name__ == "__main__":
    main()