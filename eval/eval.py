import os
import torch
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import list_models
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModelForSeq2SeqLM,
)
import re
from tqdm import tqdm
from data_preprocessing.data_processing import load_dataset_from_disk


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_seq2seq_model(model_id):
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
                else: # If no specific delimiters, try to guess the main sentence
                    # This is a last resort and might still include some unwanted text
                    first_line = gen_after_prompt.split('\n', 1)[0].strip()
                    if '.' in first_line or '!' in first_line or '?' in first_line:
                        import re
                        match = re.match(r"([^.!?]*[.!?])", first_line)
                        extracted_text = match.group(1).strip() if match else first_line
                    else:
                        extracted_text = first_line

                gen = extracted_text.replace("\"", "").strip() # Remove any leftover quotes and extra whitespace
                # --- END QWEN-SPECIFIC EXTRACTION LOGIC ---
            else:
                gen = gen[len(prompt):].strip() # Default for other causal LMs

        results.append({
            "original_toxic": orig,
            "reference_neutral": ref,
            "generated_neutral": gen
        })

    return results


def score_toxicity_and_similarity(results):
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


def evaluate_model(model_id, eval_data):
    """
    Evaluates a single model for detoxification and similarity.
    """
    results = generate_outputs(model_id, eval_data)
    results = score_toxicity_and_similarity(results)
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
    print("Searching models under TarhanE/sft-* ...")
    all_models = list_models(author="TarhanE")
    sft_models = [m.modelId for m in all_models if m.modelId.startswith("TarhanE/sft-")]

    if not sft_models:
        print("❌ No matching models found under TarhanE/sft-*.")
        # Consider adding other default models if no SFT models are found
        # sft_models = ["google/flan-t5-base"] # Example: Fallback to a single model for testing
        # print("Using a fallback model for evaluation.")
        # return # Uncomment to exit if no SFT models are strictly required

    print("📚 Loading evaluation dataset...")
    eval_ds = load_dataset_from_disk("test_dataset")
    eval_data = eval_ds.shuffle(seed=42) # Limit for fast debug

    base_models = [
        "google/flan-t5-base",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "textdetox/mbart-detox-baseline"
        "s-nlp/t5-paranmt-detox",
    ]
    all_models_to_eval = sft_models + base_models

    all_dfs = []

    for model_id in tqdm(all_models_to_eval, desc="Evaluating models"):
        try:
            print(f"🔍 Evaluating model: {model_id}")
            df = evaluate_model(model_id, eval_data)
            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to evaluate {model_id}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv("detoxification_evaluation_all_new.csv", index=False)
        print("✅ Saved all results to: detoxification_evaluation_all_new.csv")
    else:
        print("❌ No evaluations completed.")


if __name__ == "__main__":
    main()