from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "artifacts" / "model"

CONFIDENCE_THRESHOLD = 0.55

LOW_CONFIDENCE_CUTOFF = 0.30


MODEL_NAME = "xlm-roberta-base"

DATASETS = {
    "primary": "gretelai/symptom_to_diagnosis",
    "medquad": "keivalya/MedQuad-MedicalQnADataset"
}

MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
OUTPUT_DIR = "./artifacts"
