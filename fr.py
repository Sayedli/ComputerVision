#!/usr/bin/env python3
"""
Face Recognition MVP (registration, encoding, training, and live recognition)

Features:
- Register a person via webcam (captures a quick set of face crops)
- Encode all faces in dataset/ into 128-d embeddings (face_recognition)
- Train a small KNN classifier (scikit-learn)
- Recognize faces from webcam or still images

Dataset layout:
  dataset/
    Alice/
      img1.jpg, img2.jpg, ...
    Bob/
      *.jpg ...

Model artifacts:
  models/encodings.npz      # X (N x 128), y (N strings)
  models/knn.joblib         # trained KNN classifier

Usage:
  # 1) (Optional) Collect samples with webcam
  python fr.py register --name "Alice" --num 20

  # 2) Encode all dataset images to embeddings
  python fr.py encode

  # 3) Train KNN on the embeddings
  python fr.py train

  # 4) Run live webcam recognition
  python fr.py webcam

  # Or recognize a single image
  python fr.py recognize --image path/to/photo.jpg

Author: ChatGPT (GPT-5 Thinking)
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier

# Face recognition (dlib wrapper)
import face_recognition


# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
ENC_PATH = MODELS_DIR / "encodings.npz"
KNN_PATH = MODELS_DIR / "knn.joblib"

# Create folders if missing
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def list_persons(dataset_dir: Path) -> List[str]:
    """Return sorted list of person folder names in dataset/."""
    if not dataset_dir.exists():
        return []
    return sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])


def image_paths_for_person(dataset_dir: Path, person: str) -> List[Path]:
    """All images for a person."""
    pdir = dataset_dir / person
    if not pdir.exists():
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in pdir.rglob("*") if p.suffix.lower() in exts])


def draw_box_with_label(frame: np.ndarray, box: Tuple[int, int, int, int], label: str) -> None:
    """Draw a rectangle + label on frame: box=(top,right,bottom,left)."""
    top, right, bottom, left = box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 10 if top - 10 > 10 else top + 20
    cv2.rectangle(frame, (left, y - 20), (left + 6 * len(label), y), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, label, (left + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def distance_to_confidence(dist: float, threshold: float = 0.6) -> float:
    """
    Convert an embedding distance to a rough confidence in [0,1].
    Smaller distances => higher confidence.
    """
    conf = max(0.0, 1.0 - (dist / threshold))
    return float(min(1.0, conf))
