import os
import joblib
import numpy as np
import random
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder

from config import (
    MODEL_NAME,
    OUTPUT_DIR,
    EPOCHS,
    BATCH_SIZE,
    SEED,
    DATASETS
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_diagnosis_model():
    print("[INFO] Loading primary diagnosis dataset...")

    set_seed(SEED)

    ds = load_dataset(DATASETS["primary"], split="train")

    texts = ds["input_text"]
    labels = ds["output_text"]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)

    out_dir = os.path.join(OUTPUT_DIR, "train_diagnosis")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(out_dir, "label_encoder.pkl"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        enc["labels"] = batch["label"]
        return enc

    dataset = {
        "text": texts,
        "label": encoded_labels
    }

    dataset = load_dataset("json", data_files={"train": dataset})["train"]
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        seed=SEED,
        fp16=True,
        save_strategy="epoch",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )

    print("[INFO] Starting diagnosis training...")
    trainer.train()
    trainer.save_model(out_dir)

    print("[SUCCESS] Diagnosis model trained and saved.")
