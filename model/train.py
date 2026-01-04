# model/train.py

from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from data.load_data import load_primary_dataset
from config import MODEL_NAME, OUTPUT_DIR, BATCH_SIZE, EPOCHS

def train_model():
    dataset = load_primary_dataset()

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset["train"]["label_text"])

    def encode_labels(example):
        example["label"] = int(label_encoder.transform([example["label_text"]])[0])
        return example

    dataset = dataset.map(encode_labels)

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    num_labels = len(set(dataset["train"]["label"]))

    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    train_model()
