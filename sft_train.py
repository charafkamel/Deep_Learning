import os
import logging
import yaml
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from helpers import user_login
from logger import CustomLogger
from data_processing import load_dataset_from_disk


# === Tokenizer and Model ===
model_name = "google/t5-v1_1-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

##=== Preprocessing function ===
def preprocess(example):
    input_text = f"detoxify: {example['toxic']}"
    target_text = example["neutral"]
    input_enc = tokenizer(input_text, truncation=True, padding="max_length", max_length=64)
    target_enc = tokenizer(target_text, truncation=True, padding="max_length", max_length=64)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

# === Tokenize the dataset ===
dataset = load_dataset_from_disk("sft_dataset")
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset, eval_dataset = split["train"], split["test"]

# === Training Arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir="./detoxifier",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    #evaluation_strategy="epoch",
    save_total_limit=1,
    predict_with_generate=True,
    logging_dir='./logs',
)

# === Trainer ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()