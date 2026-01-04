import json
from pathlib import Path

# Resolve path safely
BASE_DIR = Path(__file__).resolve().parent.parent
KG_FILE = BASE_DIR / "knowledge_graph" / "medical_kg.json"

# Load KG once (safe)
def _load_kg():
    if not KG_FILE.exists():
        return {}

    try:
        with open(KG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

MEDICAL_KG = _load_kg()


def enrich_prediction(label: str) -> dict:
    """
    Returns conservative clinical guidance.
    Never hallucinates.
    Never crashes.
    """

    entry = MEDICAL_KG.get(label)

    if not entry:
        return {
            "description": "No structured clinical guidance available for this condition.",
            "action": "Monitor symptoms and consult a healthcare professional if they worsen."
        }

    # Defensive access
    description = entry.get(
        "description",
        "General medical condition detected."
    )

    action = entry.get(
        "action",
        "Seek medical advice if symptoms persist or worsen."
    )

    return {
        "description": description,
        "action": action
    }
