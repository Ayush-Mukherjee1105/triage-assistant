# utils/intent.py

import re

INTENT_PATTERNS = {
    "INJURY": [
        r"\bhit\b", r"\bfell\b", r"\binjury\b", r"\baccident\b",
        r"\bbleeding\b", r"\bwound\b", r"\bcut\b"
    ],
    "LIFESTYLE": [
        r"\bstress\b", r"\btired\b", r"\bfatigue\b",
        r"\bsleep\b", r"\binsomnia\b", r"\banxiety\b"
    ],
    "ILLNESS": [
        r"\bfever\b", r"\bcough\b", r"\bpain\b",
        r"\bnausea\b", r"\bvomit\b", r"\binfection\b"
    ]
}

def detect_intent(text: str) -> str:
    """
    Shallow intent classification.
    Used for contextual guidance only.
    """

    text = text.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return intent

    return "GENERAL"
