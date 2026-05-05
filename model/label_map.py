# model/label_map.py

LABEL_MAP = {
    "LABEL_0": "Acute respiratory condition",
    "LABEL_1": "Gastrointestinal issue",
    "LABEL_2": "Migraine or headache disorder",
    "LABEL_3": "General viral illness",
    "LABEL_4": "Musculoskeletal pain",
    "LABEL_5": "Upper respiratory infection",
    "LABEL_6": "Fever-related illness",
    "LABEL_7": "Digestive discomfort",
    "LABEL_8": "Tension headache",
    "LABEL_9": "Allergic reaction",
    "LABEL_10": "Dehydration symptoms",
    "LABEL_11": "Non-specific headache",
    "LABEL_12": "Ear or sinus issue",
    "LABEL_13": "Fatigue-related condition",
    "LABEL_14": "Throat irritation",
    "LABEL_15": "Neurological concern",
    "LABEL_16": "General dizziness",
    "LABEL_17": "Abdominal pain",
    "LABEL_18": "Nausea or vomiting",
    "LABEL_19": "Gastrointestinal upset",
}


def decode_label(label: str) -> str:
    """
    Convert model label to human-readable diagnosis.
    Safe fallback if label unseen.
    """
    return LABEL_MAP.get(label, label)