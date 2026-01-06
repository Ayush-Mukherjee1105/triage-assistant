import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import joblib
from config import MODEL_NAME, OUTPUT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path(OUTPUT_DIR) / "train_diagnosis"

_tokenizer = None
_model = None
_label_encoder = None


def _load():
    global _tokenizer, _model, _label_encoder
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.to(DEVICE)
        _model.eval()
        _label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")


def predict_diagnosis(text: str):
    _load()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    conf, idx = torch.max(probs, dim=1)
    label = _label_encoder.inverse_transform([idx.item()])[0]

    return label, float(conf.item())
