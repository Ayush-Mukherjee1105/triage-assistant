# model/train_duration.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
import random
import torch
from config import MODEL_NAME, OUTPUT_DIR, EPOCHS, BATCH_SIZE, SEED

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DURATION_LABELS = [
    "ACUTE",        # 0–2 days
    "SUBACUTE",     # 3–7 days
    "PROLONGED",    # 8–14 days
    "CHRONIC",      # >14 days
    "UNKNOWN"
]

label2id = {l: i for i, l in enumerate(DURATION_LABELS)}
id2label = {i: l for l, i in label2id.items()}


def map_duration(example):
    min_d = example.get("duration_days_min")
    max_d = example.get("duration_days_max")
    certainty = str(example.get("duration_certainty", "")).lower()

    if certainty == "unknown" or min_d is None or max_d is None:
        label = "UNKNOWN"
    elif max_d <= 2:
        label = "ACUTE"
    elif max_d <= 7:
        label = "SUBACUTE"
    elif max_d <= 14:
        label = "PROLONGED"
    else:
        label = "CHRONIC"

    example["label"] = label2id[label]
    return example


def train_duration_model():
    print("[INFO] Loading duration dataset...")

    ds = load_dataset(
        "json",
        data_files="data/symptom_duration.json"
    )["train"]

    ds = ds.map(map_duration)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["input_text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    ds = ds.train_test_split(test_size=0.15, seed=SEED)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(DURATION_LABELS),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/train_duration",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        seed=SEED,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer
    )

    print("[INFO] Starting duration classification training...")
    trainer.train()

    trainer.save_model(f"{OUTPUT_DIR}/train_duration")
    print("[SUCCESS] Duration model trained and saved.")
