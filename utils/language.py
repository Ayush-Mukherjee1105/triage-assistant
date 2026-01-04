from langdetect import detect_langs

# Languages we realistically support in Phase 1–3
SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
}

MIN_CHAR_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.75


def detect_language(text: str):
    """
    Robust language detection with safety overrides for short or low-confidence text.
    Returns (language_name, confidence)
    """

    text = text.strip().lower()

    # ---------- RULE 1: VERY SHORT TEXT ----------
    if len(text) < MIN_CHAR_LENGTH:
        return "English", 1.00

    try:
        detections = detect_langs(text)
        top = detections[0]
        lang_code = top.lang
        confidence = top.prob

    except Exception:
        return "English", 1.00

    # ---------- RULE 2: LOW CONFIDENCE ----------
    if confidence < CONFIDENCE_THRESHOLD:
        return "English", confidence

    # ---------- RULE 3: UNSUPPORTED LANGUAGE ----------
    if lang_code not in SUPPORTED_LANGS:
        return "English", confidence

    return SUPPORTED_LANGS[lang_code], confidence
