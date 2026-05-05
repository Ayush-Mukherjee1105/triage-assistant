import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "artifacts" / "train_duration"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_tok = None
def _load():
    global _model, _tok
    if _model is not None:
        return
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        device_map=None,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    )
    _model.eval()
    _model.to(_device)
    
def predict_duration(text: str):
    _load()

    inputs = _tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(_device)

    with torch.no_grad():
        out = _model(**inputs)

    val = float(out.logits.squeeze().cpu())

    days = max(1, int(round(val)))
    return days, days + 2
