from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier

from .paths import ENC_PATH, KNN_PATH


def load_encodings(enc_path: Path = ENC_PATH) -> Tuple[np.ndarray, np.ndarray]:
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


def train_knn(
    enc_path: Path = ENC_PATH,
    knn_path: Path = KNN_PATH,
    n_neighbors: int | None = None,
    weights: str = "distance",
    algorithm: str = "auto",
    metric: str = "euclidean",
) -> Tuple[KNeighborsClassifier, int, np.ndarray]:
    X, y = load_encodings(enc_path)
    n_samples = len(X)
    if n_samples == 0:
        raise RuntimeError("Encodings are empty; nothing to train.")

    if n_neighbors is None or n_neighbors <= 0:
        n_neighbors = max(1, int(round(np.sqrt(n_samples))))
    if n_neighbors > n_samples:
        n_neighbors = n_samples

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        metric=metric,
    )
    knn.fit(X, y)

    knn_path.parent.mkdir(parents=True, exist_ok=True)
    dump(knn, str(knn_path))

    classes = np.unique(y)
    return knn, n_neighbors, classes

