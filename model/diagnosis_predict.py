import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("artifacts/train_diagnosis")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = None
_model = None
_id2label = None


def _load():
    global _tokenizer, _model, _id2label

    if _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        device_map=None,
        low_cpu_mem_usage=False
    )

    _model.to(_device)
    _model.eval()

    _id2label = _model.config.id2label


def predict_diagnosis(text: str):
    _load()

    with torch.no_grad():
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs = {k: v.to(_device) for k, v in inputs.items()}

        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        idx = int(torch.argmax(probs).cpu().numpy())
        conf = float(probs[idx].cpu().numpy())

    label = _id2label.get(idx, "UNKNOWN")

    return label, round(conf, 3)
