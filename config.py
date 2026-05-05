from pathlib import Path
import random
import numpy as np
import torch

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
DIAG_MODEL_DIR = ARTIFACTS_DIR / "train_diagnosis"
RED_FLAG_MODEL_DIR = ARTIFACTS_DIR / "train_red_flag"
DURATION_MODEL_DIR = ARTIFACTS_DIR / "train_duration"

# -------------------------
# Model
# -------------------------
MODEL_NAME = "xlm-roberta-base"

# -------------------------
# Confidence thresholds
# -------------------------
CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_CUTOFF = 0.30

# -------------------------
# Dataset registry
# -------------------------
DATASETS = {
    "primary": "gretelai/symptom_to_diagnosis",
    "medquad": "keivalya/MedQuad-MedicalQnADataset",

    # Clinical NER
    "ncbi_disease": "ncbi_disease",
    "bionlp_symptoms": "bionlp2013_cg",

    # Dialog grounding
    "medical_dialog": "medical_dialog"
}

USE_DATASETS = {
    "primary": True,
    "medquad": True,
    "ncbi_disease": True,
    "bionlp_symptoms": True,
    "medical_dialog": True
}

# -------------------------
# Training toggles
# -------------------------
OUTPUT_DIR = "artifacts"

TRAIN_DIAGNOSIS = False
TRAIN_RED_FLAG = True
TRAIN_DURATION = False

USE_MEDQUAD_FOR_TRAINING = True
USE_MEDICAL_DIALOG_FOR_TRAINING = False  

# -------------------------
# Training hyperparams
# -------------------------
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 5   # ↑ increase from 3 → 5 for better convergence
