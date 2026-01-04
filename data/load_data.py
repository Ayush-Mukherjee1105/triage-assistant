# data/load_data.py

from datasets import load_dataset
from config import DATASETS

def load_primary_dataset():
    """
    Loads and normalizes gretelai/symptom_to_diagnosis
    """
    ds = load_dataset(DATASETS["primary"])

    def map_fn(example):
        return {
            "text": example["input_text"],
            "label_text": example["output_text"]
        }

    ds = ds.map(map_fn)
    ds = ds.remove_columns(
        [col for col in ds["train"].column_names if col not in ["text", "label_text"]]
    )

    return ds


def load_medquad():
    """
    Loaded for knowledge graph / grounding (NOT training)
    """
    ds = load_dataset(DATASETS["medquad"], split="train")
    return ds
