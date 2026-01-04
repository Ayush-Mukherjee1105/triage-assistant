# model/xai.py

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

MODEL_PATH = "artifacts/model"

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_PATH,
    output_attentions=True
)
model.eval()


def explain(text, top_k=10):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Last layer attention, averaged across heads
    attentions = outputs.attentions[-1].mean(dim=1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    scores = attentions.tolist()

    token_scores = list(zip(tokens, scores))
    token_scores.sort(key=lambda x: x[1], reverse=True)

    return token_scores[:top_k]
