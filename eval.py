#!/usr/bin/env python
# -- coding: utf-8 --

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
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from tqdm import tqdm
from data_processing import load_dataset_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_t5(model_id):
    config = AutoConfig.from_pretrained(model_id)
    return config.model_type == "t5"


def build_prompt(model_id, toxic):
    if is_t5(model_id):
        return f"detoxify: {toxic}"
    else:
        return f"detoxify: {toxic}\nNeutral:"


def generate_outputs(model_id, eval_data, max_length=64):
    is_t5_model = is_t5(model_id)

    if is_t5_model:
        tokenizer = T5Tokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
        generate_fn = lambda m, t: m.generate(**t, max_length=max_length, num_beams=4, early_stopping=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        generate_fn = lambda m, t: m.generate(**t, max_new_tokens=max_length)

    model.eval()
    results = []

    for ex in eval_data:
        orig = ex["toxic"]
        ref = ex["neutral"]
        prompt = build_prompt(model_id, orig)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            out_ids = generate_fn(model, enc)

        gen = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if not is_t5_model:
            gen = gen[len(prompt):].strip()

        results.append({
            "original_toxic": orig,
            "reference_neutral": ref,
            "generated_neutral": gen
        })
    return results


def score_toxicity_and_similarity(results):
    tox_tok = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    tox_mod = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device).eval()
    label2id = tox_mod.config.label2id
    tox_label = label2id.get("toxicity", list(label2id.values())[0])

    sim_tok = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    sim_mod = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device).eval()

    origs = [r["original_toxic"] for r in results]
    refs  = [r["reference_neutral"] for r in results]
    gens  = [r["generated_neutral"] for r in results]

    with torch.no_grad():
        tox_inputs = tox_tok(origs + refs + gens, return_tensors="pt", truncation=True, padding=True).to(device)
        tox_logits = tox_mod(**tox_inputs).logits
        tox_probs = torch.sigmoid(tox_logits)

        sim_inputs = sim_tok(origs + refs + gens, return_tensors="pt", truncation=True, padding=True).to(device)
        emb_all = sim_mod(**sim_inputs).pooler_output

    n = len(results)
    tox_orig, tox_ref, tox_gen = tox_probs[:n], tox_probs[n:2*n], tox_probs[2*n:]
    sim_orig, sim_ref, sim_gen = emb_all[:n], emb_all[n:2*n], emb_all[2*n:]

    for i, r in enumerate(results):
        r["toxicity_orig"] = float(tox_orig[i][tox_label])
        r["toxicity_ref"] = float(tox_ref[i][tox_label])
        r["toxicity_gen"] = float(tox_gen[i][tox_label])
        r["similarity_orig_ref"] = float(F.cosine_similarity(sim_orig[i], sim_ref[i], dim=0).item())
        r["similarity_orig_gen"] = float(F.cosine_similarity(sim_orig[i], sim_gen[i], dim=0).item())
    return results


def evaluate_model(model_id, eval_data):
    
    results = generate_outputs(model_id, eval_data)
    results = score_toxicity_and_similarity(results)
    print(f"✅ Evaluation completed for model: {model_id}")
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
    print("🔍 Searching models under TarhanE/sft-* ...")
    all_models = list_models(author="TarhanE")
    sft_models = [m.modelId for m in all_models if m.modelId.startswith("TarhanE/sft-")]

    if not sft_models:
        print("❌ No matching models found.")
        return

    print("📚 Loading evaluation dataset...")
    eval_ds = load_dataset_from_disk("test_dataset")
    eval_data = eval_ds.shuffle(seed=42)
    
    # Add base models for comparison
    base_models = [
        "google/t5-v1_1-base",
        "Qwen/Qwen3-0.6B"
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

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv("detoxification_evaluation_all.csv", index=False)
        print("✅ Saved all results to: detoxification_evaluation_all.csv")
    else:
        print("❌ No evaluations completed.")


if __name__ == "__main__":
    main()
