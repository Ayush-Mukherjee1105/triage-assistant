# utils/intent.py

def detect_intent(text: str) -> str:
    t = text.lower()

    if any(k in t for k in ["hit", "injury", "fell", "accident", "bleeding"]):
        return "INJURY"

    if any(k in t for k in ["sleep", "stress", "fatigue", "tired"]):
        return "LIFESTYLE"

    if any(k in t for k in ["fever", "pain", "vomit", "infection", "cough"]):
        return "ILLNESS"

    return "GENERAL"
