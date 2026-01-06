# model/xai.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_NAME, OUTPUT_DIR

_model = None
_tokenizer = None

def _load():
    global _model, _tokenizer
    if _model is None:
        model_dir = f"{OUTPUT_DIR}/train_diagnosis"
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        _model.eval()

def explain(text: str, top_k=5):
    """
    Token-level importance via gradient × input (simple, defensible)
    """
    _load()

    inputs = _tokenizer(text, return_tensors="pt", truncation=True)
    inputs.requires_grad_(True)

    outputs = _model(**inputs)
    logits = outputs.logits
    pred_idx = logits.argmax(dim=-1)

    score = logits[0, pred_idx]
    score.backward()

    grads = inputs["input_ids"].grad.abs().sum(dim=0)
    tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    token_scores = list(zip(tokens, grads.tolist()))
    token_scores = sorted(token_scores, key=lambda x: x[1], reverse=True)

    return token_scores[:top_k]
