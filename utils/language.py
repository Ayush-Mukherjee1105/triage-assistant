# utils/language.py

from langdetect import detect_langs, DetectorFactory
import re

DetectorFactory.seed = 42  # reproducibility


# ----------------------------
# SCRIPT DETECTION (STRONG)
# ----------------------------

SCRIPT_RANGES = {
    "Hindi": [(0x0900, 0x097F)],       # Devanagari
    "Bengali": [(0x0980, 0x09FF)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Japanese": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)],
    "Chinese": [(0x4E00, 0x9FFF)],
    "Korean": [(0xAC00, 0xD7AF)],
    "Thai": [(0x0E00, 0x0E7F)],
}


def _detect_by_script(text: str):
    for char in text:
        code = ord(char)
        for lang, ranges in SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code <= end:
                    return lang, 0.95
    return None


# ----------------------------
# ROMANIZED INDIAN HEURISTICS
# ----------------------------

ROMANIZED_HINTS = {
    "Hindi": [
        r"\bhai\b", r"\bnahi\b", r"\bdard\b", r"\buk\b", r"\bkyun\b",
        r"\bmujhe\b", r"\bchahiye\b"
    ],
    "Bengali": [
        r"\bache\b", r"\bamar\b", r"\bmatha\b", r"\bkhub\b"
    ],
    "Tamil": [
        r"\bnalla\b", r"\brendu\b", r"\bvalikkuthu\b"
    ],
}


def _detect_romanized(text: str):
    text = text.lower()
    for lang, patterns in ROMANIZED_HINTS.items():
        for pat in patterns:
            if re.search(pat, text):
                return lang + " (Romanized)", 0.65
    return None


# ----------------------------
# MAIN API
# ----------------------------

def detect_language(text: str):
    """
    Robust language detection for short, informal, Indian/Asian text.
    Returns (language_name, confidence).
    """

    if not text or len(text.strip()) < 3:
        return "Unknown", 0.0

    # 1️⃣ Script detection
    script_result = _detect_by_script(text)
    if script_result:
        return script_result

    # 2️⃣ Romanized Indian detection
    romanized_result = _detect_romanized(text)
    if romanized_result:
        return romanized_result

    # 3️⃣ Statistical fallback
    try:
        langs = detect_langs(text)
        top = langs[0]
        lang = top.lang

        LANG_MAP = {
            "en": "English",
            "hi": "Hindi",
            "bn": "Bengali",
            "ta": "Tamil",
            "te": "Telugu",
            "ml": "Malayalam",
            "ja": "Japanese",
            "zh": "Chinese",
            "ko": "Korean",
            "th": "Thai",
        }

        return LANG_MAP.get(lang, lang), min(top.prob, 0.85)

    except Exception:
        return "Unknown", 0.0
