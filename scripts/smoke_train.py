#!/usr/bin/env python3
"""
Smoke test for training (no external I/O).

Creates synthetic 128-D embeddings for two classes, trains a KNN
using fr.train.train_knn_from_arrays, and prints a simple accuracy.
"""

from __future__ import annotations

import numpy as np

from fr.train import train_knn_from_arrays


def main() -> None:
    rng = np.random.default_rng(42)
    n_per_class = 20
    dim = 128

    # Two Gaussian blobs in 128-D
    mu_a = np.zeros(dim)
    mu_b = np.ones(dim) * 1.5
    X_a = rng.normal(loc=mu_a, scale=0.2, size=(n_per_class, dim)).astype(np.float32)
    X_b = rng.normal(loc=mu_b, scale=0.2, size=(n_per_class, dim)).astype(np.float32)
    X = np.vstack([X_a, X_b])
    y = np.array(["A"] * n_per_class + ["B"] * n_per_class)

    # Shuffle
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    model, k_used, classes = train_knn_from_arrays(X, y, n_neighbors=3)

    preds = model.predict(X)
    acc = (preds == y).mean()
    print(f"Classes: {list(classes)}  k={k_used}  accuracy={acc:.3f}")


if __name__ == "__main__":
    main()

