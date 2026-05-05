# utils/roman_response.py

import re

# -------------------------------
# Hinglish phrase map
# -------------------------------

HINGLISH_PHRASES = {
    r"\brest\b": "aaram karein",
    r"\bstay hydrated\b": "paani peete rahein",
    r"\bmanage stress\b": "stress kam rakhein",
    r"\bconsult a doctor\b": "doctor se salah lein",
    r"\bseek medical advice\b": "doctor se salah lein",
    r"\bmonitor symptoms\b": "lakshan monitor karein",
}

# -------------------------------
# Bengalish phrase map
# -------------------------------

BENGALISH_PHRASES = {
    r"\brest\b": "bishram nin",
    r"\bstay hydrated\b": "paani khete thakun",
    r"\bmanage stress\b": "stress kom rakhun",
    r"\bconsult a doctor\b": "doctor er kache jaan",
    r"\bseek medical advice\b": "doctor er kache jaan",
    r"\bmonitor symptoms\b": "lokkhon monitor korun",
}


def _apply_map(text: str, mapping: dict) -> str:
    out = text
    for pattern, repl in mapping.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def to_hinglish(text: str) -> str:
    return _apply_map(text, HINGLISH_PHRASES)


def to_bengalish(text: str) -> str:
    return _apply_map(text, BENGALISH_PHRASES)