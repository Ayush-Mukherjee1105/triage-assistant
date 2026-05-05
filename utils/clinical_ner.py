# utils/clinical_ner.py

import re

SYMPTOMS = [
    "headache", "dard", "pain", "fever", "cough", "breath", "saans",
    "dizziness", "ghurche", "vomit", "nausea", "chest", "seene",
]

BODY_PARTS = [
    "head", "sir", "matha", "chest", "seena", "pet", "stomach", "abdomen"
]

def extract_entities(text: str) -> dict:
    t = text.lower()

    symptoms = [s for s in SYMPTOMS if s in t]
    body_parts = [b for b in BODY_PARTS if b in t]

    return {
        "symptoms": list(set(symptoms)),
        "body_parts": list(set(body_parts)),
        "conditions": [],
        "durations": []
    }
