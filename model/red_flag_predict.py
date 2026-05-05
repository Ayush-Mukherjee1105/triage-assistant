# model/red_flag_predict.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(
    "artifacts/train_red_flag",
    fix_mistral_regex=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    "artifacts/train_red_flag"
).to(DEVICE)
model.eval()
def predict_red_flag(text: str):
    """
    Returns:
        (is_red_flag: bool, label: str, confidence: float)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    label_id = pred.item()
    confidence = float(conf.item())
    label_map = {
        0: "NO_RED_FLAG",
        1: "RED_FLAG",
    }
    label = label_map.get(label_id, "NO_RED_FLAG")
    is_red = label == "RED_FLAG"
    return is_red, label, confidence