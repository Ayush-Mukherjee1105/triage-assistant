# model/train_red_flag.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
import torch
import os
import joblib

from config import MODEL_NAME, OUTPUT_DIR, EPOCHS, BATCH_SIZE, SEED

def train_red_flag_model():
    print("[INFO] Loading red-flag dataset...")

    ds = load_dataset(
        "csv",
        data_files="data/clinical_red_flag_triage.csv"
    )["train"]

    # --- HANDLE NULL LABELS ---
    def clean(example):
        if example["red_flag_type"] is None or example["red_flag_type"] == "":
            example["red_flag_type"] = "NO_REDFLAG"
        return example

    ds = ds.map(clean)

    # --- LABEL ENCODING ---
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(ds["red_flag_type"])

    ds = ds.add_column("label", labels)

    # --- TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["input_text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    ds = ds.map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # --- MODEL ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_)
    )

    # --- TRAINING ---
    out_dir = f"{OUTPUT_DIR}/train_red_flag"
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        seed=SEED,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds
    )

    print("[INFO] Starting red-flag training...")
    trainer.train()

    # --- SAVE EVERYTHING ---
    trainer.save_model(out_dir)
    joblib.dump(label_encoder, f"{out_dir}/label_encoder.pkl")

    print("[SUCCESS] Red-flag model trained and saved.")

if __name__ == "__main__":
    train_red_flag_model()
