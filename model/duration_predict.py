# model/duration_predict.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from config import MODEL_NAME

MODEL_DIR = Path("artifacts/train_duration")

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()


def predict_duration(text: str):
    _load()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][idx].item()

    label = _model.config.id2label[str(idx)]
    return label, confidence
