# model/red_flag_predict.py

import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import OUTPUT_DIR, MODEL_NAME

MODEL_DIR = f"{OUTPUT_DIR}/train_red_flag"

_tokenizer = None
_model = None
_label_encoder = None

def _load_model():
    global _tokenizer, _model, _label_encoder

    if _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.eval()

    _label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

    if torch.cuda.is_available():
        _model.cuda()

def predict_red_flag(text: str):
    """
    Returns:
      (is_red_flag: bool, label: str, confidence: float)
    """
    _load_model()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    conf, idx = torch.max(probs, dim=-1)
    label = _label_encoder.inverse_transform([idx.item()])[0]

    return label != "NO_REDFLAG", label, float(conf.item())
