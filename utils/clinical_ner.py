from transformers import pipeline

# Load once (GPU-aware)
_ner_pipeline = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple",
    device=0  # uses CUDA if available
)

def extract_clinical_entities(text: str) -> dict:
    """
    Extracts clinical entities and normalizes them
    for downstream severity / triage logic.
    """
    entities = {
        "symptoms": [],
        "conditions": [],
        "durations": [],
        "body_parts": [],
    }

    if not text or not text.strip():
        return entities

    results = _ner_pipeline(text)

    for r in results:
        label = r.get("entity_group", "").lower()
        value = r.get("word", "").lower()

        if label in ["sign_symptom", "symptom"]:
            entities["symptoms"].append(value)

        elif label in ["disease", "diagnosis"]:
            entities["conditions"].append(value)

        elif label in ["duration", "time"]:
            entities["durations"].append(value)

        elif label in ["body_part", "anatomy"]:
            entities["body_parts"].append(value)

    return entities
