# utils/language.py

"""
Multilingual language + script detector

Returns one of:
- English
- Hindi
- Bengali
- Hinglish
- Bengalish
"""

import re


# -----------------------------
# Script detectors
# -----------------------------

def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def _has_bengali(text: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in text)


# -----------------------------
# Roman heuristics
# -----------------------------

HINGLISH_HINTS = [
    "mujhe", "bukhar", "dard", "sir", "pet",
    "raha", "rahi", "hai", "ho raha", "kal"
]

BENGALISH_HINTS = [
    "amar", "matha", "betha", "hocche", "pet",
    "jhor", "lagche", "ache", "kemon", "betha hocche"
]


def _roman_score(text: str, vocab: list) -> int:
    text = text.lower()
    return sum(1 for w in vocab if w in text)


# -----------------------------
# MAIN DETECTOR
# -----------------------------

def normalize_text(text: str) -> str:
    return text.strip()


def detect_language(text: str) -> str:
    """
    Returns:
    English | Hindi | Bengali | Hinglish | Bengalish
    """

    if not text:
        return "English"

    # Script wins first
    if _has_devanagari(text):
        return "Hindi"

    if _has_bengali(text):
        return "Bengali"

    # Roman detection
    hi_score = _roman_score(text, HINGLISH_HINTS)
    bn_score = _roman_score(text, BENGALISH_HINTS)

    if bn_score >= 2 and bn_score > hi_score:
        return "Bengalish"

    if hi_score >= 2:
        return "Hinglish"

    return "English"