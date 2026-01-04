import re

# Minimal but strong medical signal vocabulary
MEDICAL_KEYWORDS = [
    "pain", "ache", "hurt", "injury", "bleeding", "fever",
    "vomit", "vomiting", "nausea", "dizzy", "dizziness",
    "headache", "head", "stomach", "chest", "breath",
    "cough", "cold", "infection", "sick", "ill",
    "swelling", "fracture", "burn", "rash", "itch",
    "diarrhea", "constipation", "sleep", "insomnia"
]

def is_medical_text(text: str) -> bool:
    """
    Gatekeeper:
    Returns True ONLY if text has medical relevance.
    """
    text = text.lower()

    # Very short text is usually not medical
    if len(text.split()) < 2:
        return False

    for kw in MEDICAL_KEYWORDS:
        if re.search(rf"\b{kw}\b", text):
            return True

    return False
