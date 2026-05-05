import os, json, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from config import MAX_LENGTH, BATCH_SIZE, EPOCHS
DATA_PATH = "data/symptom_duration.json"
OUT_DIR = "artifacts/train_duration"
BASE_MODEL = "xlm-roberta-base"
class DurationDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.texts = [r["input_text"] for r in rows]
        self.labels = [float(r["duration_log"]) for r in rows]
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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
def train_duration_model():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = json.load(open(DATA_PATH, "r", encoding="utf8"))
    rows = [r for r in rows if "input_text" in r and "duration_log" in r]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        problem_type="regression",
    )
    dataset = DurationDataset(rows, tokenizer)
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
    print("✅ Duration model trained")
if __name__ == "__main__":
    train_duration_model()
