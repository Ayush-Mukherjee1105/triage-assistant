# utils/diagnosis_sanity.py

RESPIRATORY_HINTS = [
    "cough",
    "khansi",
    "saans",
    "breath",
    "breathing",
    "cold",
    "sardi",
]

GASTRO_HINTS = [
    "vomit",
    "vomiting",
    "ultee",
    "pet",
    "stomach",
    "nausea",
]

HEADACHE_HINTS = [
    "headache",
    "sar dard",
    "migraine",
    "matha",
]


def correct_diagnosis(label: str, text: str) -> str:
    """
    Lightweight clinical sanity correction.
    Does NOT change confidence.
    Safe for research use.
    """

    t = text.lower()

    # -------------------------------------------------
    # Respiratory override
    # -------------------------------------------------
    if any(k in t for k in RESPIRATORY_HINTS):
        if label in ["LABEL_2", "LABEL_11"]:  # headache cluster
            return "LABEL_5"  # Upper respiratory infection

    # -------------------------------------------------
    # Gastro override
    # -------------------------------------------------
    if any(k in t for k in GASTRO_HINTS):
        if label not in ["LABEL_18", "LABEL_19"]:
            return "LABEL_18"  # Nausea or vomiting

    # -------------------------------------------------
    # Headache reinforcement (prevents over-correction)
    # -------------------------------------------------
    if any(k in t for k in HEADACHE_HINTS):
        if label == "LABEL_5":
            return "LABEL_11"

    return label