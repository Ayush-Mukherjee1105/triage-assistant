from datasets import load_dataset
from config import DATASETS, USE_DATASETS
import json
import pandas as pd
from pathlib import Path


# -------------------------
# DATASET 1 — DIAGNOSIS
# -------------------------
def load_primary_dataset():
    """
    gretelai/symptom_to_diagnosis
    Used ONLY for diagnosis classification
    """
    ds = load_dataset(DATASETS["primary"])

    def map_fn(example):
        return {
            "text": example["input_text"],
            "label_text": example["output_text"]
        }

    ds = ds.map(map_fn)
    ds = ds.remove_columns(
        [c for c in ds["train"].column_names if c not in ["text", "label_text"]]
    )

    return ds


# -------------------------
# DATASET 2 — MEDQUAD (GROUNDING ONLY)
# -------------------------
def load_medquad():
    """
    Used ONLY for knowledge graph / grounding
    NOT used for training
    """
    return load_dataset(DATASETS["medquad"], split="train")


# -------------------------
# DATASET 5 — RED FLAG TRIAGE
# -------------------------
def load_red_flag_dataset():
    """
    clinical_red_flag_triage.csv
    Used to TRAIN red-flag classifier
    """
    path = Path("data/clinical_red_flag_triage.csv")
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        records.append({
            "text": row["input_text"],
            "red_flag": int(row["red_flag"]),
            "red_flag_type": row["red_flag_type"],
            "reason": row["clinical_reasoning"]
        })

    return records


# -------------------------
# DATASET 6 — SYMPTOM DURATION
# -------------------------
def load_duration_dataset():
    """
    symptom_duration.json
    Used to TRAIN duration extraction model
    """
    path = Path("data/symptom_duration.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        records.append({
            "text": item["input_text"],
            "duration_phrase": item["duration_phrase"],
            "days_min": item["duration_days_min"],
            "days_max": item["duration_days_max"],
            "certainty": item["duration_certainty"],
            "note": item["clinical_note"]
        })

    return records


# -------------------------
# AUXILIARY DATASETS (NO TRAINING)
# -------------------------
def load_auxiliary_datasets():
    """
    NER + medical dialog (support only)
    """
    aux = {}

    if USE_DATASETS.get("ncbi_disease"):
        aux["ncbi_disease"] = load_dataset(DATASETS["ncbi_disease"])

    if USE_DATASETS.get("bionlp_symptoms"):
        aux["bionlp_symptoms"] = load_dataset(DATASETS["bionlp_symptoms"])

    if USE_DATASETS.get("medical_dialog"):
        aux["medical_dialog"] = load_dataset(
            DATASETS["medical_dialog"],
            "english"
        )

    return aux
