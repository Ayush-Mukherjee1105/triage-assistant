import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from config import MAX_LENGTH, BATCH_SIZE, EPOCHS
DATA_PATH = "data/clinical_red_flag_triage.csv"
OUT_DIR = "artifacts/train_red_flag"
BASE_MODEL = "xlm-roberta-base"
class RedFlagDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["input_text"].astype(str).tolist()
        self.labels = df["red_flag"].astype(int).tolist()
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
def train_red_flag_model():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["input_text", "red_flag"]).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    dataset = RedFlagDataset(df, tokenizer)
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("✅ Red flag model trained")
if __name__ == "__main__":
    train_red_flag_model()
