#!/usr/bin/env python
# -- coding: utf-8 --

import os
import yaml
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

def generate_outputs(model_id, eval_data, max_length=64):
    if is_t5(model_id):
        tokenizer = T5Tokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
        prompt_fn = lambda x: f"detoxify: {x}"
        generate_fn = lambda m, t: m.generate(**t, max_length=max_length, num_beams=4, early_stopping=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt_fn = lambda x: x
        generate_fn = lambda m, t: m.generate(**t, max_new_tokens=max_length)

    model.eval()
    results = []

    for ex in eval_data:
        orig = ex["toxic"]
        ref = ex["neutral"]
        inp = prompt_fn(orig)
        enc = tokenizer(inp, return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            out_ids = generate_fn(model, enc)
        gen = tokenizer.decode(out_ids[0], skip_special_tokens=True)

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

    texts = [r["original_toxic"] for r in results] + \
            [r["reference_neutral"] for r in results] + \
            [r["generated_neutral"] for r in results]

    with torch.no_grad():
        tox_inputs = tox_tok(texts, return_tensors="pt", truncation=True, padding=True).to(device)
        tox_logits = tox_mod(**tox_inputs).logits
        tox_probs = torch.sigmoid(tox_logits)

        sim_inputs = sim_tok(texts, return_tensors="pt", truncation=True, padding=True).to(device)
        emb_all = sim_mod(**sim_inputs).pooler_output

    n = len(results)
    for i, r in enumerate(results):
        r["toxicity_ref"] = float(tox_probs[n+i][tox_label])
        r["toxicity_gen"] = float(tox_probs[2*n+i][tox_label])
        r["similarity_orig_ref"] = float(F.cosine_similarity(emb_all[i], emb_all[n+i], dim=0).item())
        r["similarity_orig_gen"] = float(F.cosine_similarity(emb_all[i], emb_all[2*n+i], dim=0).item())
    return results

def score_fluency(results):
    lm_tok = AutoTokenizer.from_pretrained("gpt2")
    if lm_tok.pad_token is None:
        lm_tok.pad_token = lm_tok.eos_token
    lm_mod = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

    for r in results:
        for key in ["reference_neutral", "generated_neutral"]:
            enc = lm_tok(r[key], return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                loss = lm_mod(**enc, labels=enc["input_ids"]).loss.item()
            r[f"fluency_{'ref' if key == 'reference_neutral' else 'gen'}"] = loss
    return results

def evaluate_model(model_id, eval_data):
    results = generate_outputs(model_id, eval_data)
    results = score_toxicity_and_similarity(results)
    results = score_fluency(results)

    df = pd.DataFrame(results)[[
        "original_toxic",
        "reference_neutral",
        "toxicity_ref",
        "similarity_orig_ref",
        "fluency_ref",
        "generated_neutral",
        "toxicity_gen",
        "similarity_orig_gen",
        "fluency_gen"
    ]]
    return df

def main():
    print("🔍 Searching models under TarhanE/sft-* ...")
    all_models = list_models(author="TarhanE")
    sft_models = [m.modelId for m in all_models if m.modelId.startswith("TarhanE/sft-")]

    if not sft_models:
        print("❌ No matching models found.")
        return

    print(f"📚 Loading evaluation dataset...")

    eval_ds  = load_dataset_from_disk("test_dataset")
    n_trials = 20
    eval_data  = eval_ds.shuffle(seed=42).select(range(n_trials))

    all_dfs = []

    for model_id in tqdm(sft_models, desc="Evaluating models"):
        try:
            df = evaluate_model(model_id, eval_data)
            df.insert(0, "model_id", model_id)
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
