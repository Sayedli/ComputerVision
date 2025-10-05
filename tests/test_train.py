from __future__ import annotations

import numpy as np
import pytest

from fr.train import train_knn_from_arrays


def make_blobs(n_per_class: int = 20, dim: int = 128, seed: int = 123):
    rng = np.random.default_rng(seed)
    mu_a = np.zeros(dim)
    mu_b = np.ones(dim) * 1.5
    X_a = rng.normal(loc=mu_a, scale=0.2, size=(n_per_class, dim)).astype(np.float32)
    X_b = rng.normal(loc=mu_b, scale=0.2, size=(n_per_class, dim)).astype(np.float32)
    X = np.vstack([X_a, X_b])
    y = np.array(["A"] * n_per_class + ["B"] * n_per_class)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def test_train_knn_from_arrays_basic():
    X, y = make_blobs()
    model, k_used, classes = train_knn_from_arrays(X, y, n_neighbors=3)
    assert set(classes.tolist()) == {"A", "B"}
    assert k_used == 3
    preds = model.predict(X)
    acc = (preds == y).mean()
    assert acc >= 0.95


def test_train_knn_from_arrays_auto_k():
    X, y = make_blobs(n_per_class=5)  # total 10 samples, sqrt(10) ~ 3
    model, k_used, classes = train_knn_from_arrays(X, y, n_neighbors=None)
    assert 1 <= k_used <= len(X)
    assert isinstance(k_used, int)


def test_train_knn_from_arrays_shape_errors():
    X = np.zeros((5, 127), dtype=np.float32)  # wrong dim
    y = np.array(["A"] * 5)
    with pytest.raises(ValueError):
        train_knn_from_arrays(X, y)

    X2 = np.zeros((5, 128), dtype=np.float32)
    y2 = np.array(["A"] * 4)
    with pytest.raises(ValueError):
        train_knn_from_arrays(X2, y2)

