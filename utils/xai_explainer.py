import torch
import matplotlib
matplotlib.use("Agg")  

import matplotlib.pyplot as plt
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import OUTPUT_DIR



_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer

    if _model is None:
        model_dir = f"{OUTPUT_DIR}/train_diagnosis"

        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        _model.eval()


def _merge_tokens(tokens, scores):
    """
    Merge subword tokens into real words
    """
    words = []
    current_word = ""
    current_score = 0

    for token, score in zip(tokens, scores):

        if token in ["<s>", "</s>"]:
            continue

        if token.startswith("▁"):
            if current_word:
                words.append((current_word, current_score))

            current_word = token.replace("▁", "")
            current_score = score
        else:
            current_word += token
            current_score += score

    if current_word:
        words.append((current_word, current_score))

    return words


def explain(text: str, top_k=5):

    _load()

    inputs = _tokenizer(text, return_tensors="pt", truncation=True)

    input_ids = inputs["input_ids"]

    embeddings = _model.get_input_embeddings()(input_ids)
    embeddings.retain_grad()

    outputs = _model(inputs_embeds=embeddings)

    logits = outputs.logits
    pred_idx = logits.argmax(dim=-1)

    score = logits[0, pred_idx]
    score.backward()

    grads = embeddings.grad

    token_importance = (grads * embeddings).abs().sum(dim=-1)[0]

    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = token_importance.detach().tolist()

    merged = _merge_tokens(tokens, scores)

    merged = sorted(merged, key=lambda x: x[1], reverse=True)

    return merged[:top_k]


def visualize_explanation(text: str):

    tokens = explain(text, top_k=8)

    words = [t[0] for t in tokens]
    scores = [t[1] for t in tokens]

    os.makedirs("static", exist_ok=True)

    path = "static/xai_plot.png"

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.bar(words, scores)

    plt.title("Token Importance (XAI)")
    plt.xlabel("Tokens")
    plt.ylabel("Importance")

    plt.tight_layout()
    plt.savefig(path)
    plt.close("all") 

    return path