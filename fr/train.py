from __future__ import annotations

"""Training utilities for the face recognition MVP.

Functions here handle loading saved encodings and training a KNN classifier.
Includes a pure helper that trains directly from in‑memory arrays for smoke tests.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier

from .paths import ENC_PATH, KNN_PATH


def load_encodings(enc_path: Path = ENC_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from a compressed NPZ file.

    Returns X (N,128) float array and y (N,) labels array.
    """
    if not enc_path.exists():
        raise FileNotFoundError(f"Encodings file not found: {enc_path}. Run 'encode' first.")
    data = np.load(str(enc_path), allow_pickle=True)
    X = data["X"]
    y = data["y"]
    if X.ndim != 2 or X.shape[1] != 128:
        raise ValueError("Encodings X must have shape (N, 128).")
    if len(X) != len(y):
        raise ValueError("Encodings X and labels y have mismatched lengths.")
    return X, y


def _resolve_k(n_neighbors: int | None, n_samples: int) -> int:
    if n_neighbors is None or n_neighbors <= 0:
        n_neighbors = max(1, int(round(np.sqrt(n_samples))))
    if n_neighbors > n_samples:
        n_neighbors = n_samples
    return n_neighbors


def train_knn_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int | None = None,
    weights: str = "distance",
    algorithm: str = "auto",
    metric: str = "euclidean",
) -> Tuple[KNeighborsClassifier, int, np.ndarray]:
    """Train a KNN model from in‑memory arrays without touching the filesystem.

    Returns the fitted model, the k used, and the unique classes.
    """
    if X.ndim != 2 or X.shape[1] != 128:
        raise ValueError("X must have shape (N, 128)")
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")

    n_samples = len(X)
    if n_samples == 0:
        raise RuntimeError("No samples provided.")

    k = _resolve_k(n_neighbors, n_samples)
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        algorithm=algorithm,
        metric=metric,
    )
    knn.fit(X, y)
    classes = np.unique(y)
    return knn, k, classes


def train_knn(
    enc_path: Path = ENC_PATH,
    knn_path: Path = KNN_PATH,
    n_neighbors: int | None = None,
    weights: str = "distance",
    algorithm: str = "auto",
    metric: str = "euclidean",
) -> Tuple[KNeighborsClassifier, int, np.ndarray]:
    """Train a KNN model from saved encodings and persist it to disk."""
    X, y = load_encodings(enc_path)
    knn, k, classes = train_knn_from_arrays(
        X,
        y,
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        metric=metric,
    )
    knn_path.parent.mkdir(parents=True, exist_ok=True)
    dump(knn, str(knn_path))
    return knn, k, classes
