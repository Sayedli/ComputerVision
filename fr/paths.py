from __future__ import annotations

from pathlib import Path

# Project root is the repository root (one level up from this package)
ROOT = Path(__file__).resolve().parent.parent

# Data/model directories
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
ENC_PATH = MODELS_DIR / "encodings.npz"
KNN_PATH = MODELS_DIR / "knn.joblib"

# Ensure directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

