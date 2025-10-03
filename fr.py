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


# ----------------------------
# Encode dataset -> embeddings
# ----------------------------
def encode_dataset(
    dataset_dir: Path = DATASET_DIR,
    enc_path: Path = ENC_PATH,
    model: str = "hog",
    upsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Walk `dataset_dir`, compute 128-D embeddings for each detected face, and save to `enc_path`.

    Returns: (X, y, n_images, n_faces)
    - X: float32 array of shape (N, 128)
    - y: array of labels (dtype=object or <U)
    - n_images: total images scanned
    - n_faces: total faces encoded
    """
    persons = list_persons(dataset_dir)
    if not persons:
        raise RuntimeError(f"No person folders found in {dataset_dir}. Add data or run 'register'.")

    all_paths: List[Tuple[str, Path]] = []
    for person in persons:
        for p in image_paths_for_person(dataset_dir, person):
            all_paths.append((person, p))

    if not all_paths:
        raise RuntimeError(f"No images found under {dataset_dir}.")

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    n_faces = 0

    for person, path in tqdm(all_paths, desc="Encoding", unit="img"):
        img = cv2.imread(str(path))
        if img is None:
            tqdm.write(f"[warn] Failed to read image: {path}")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            boxes = face_recognition.face_locations(
                rgb, number_of_times_to_upsample=upsample, model=model
            )
            if not boxes:
                tqdm.write(f"[skip] No face: {path}")
                continue
            encs = face_recognition.face_encodings(rgb, boxes, num_jitters=1)
        except Exception as e:
            tqdm.write(f"[error] {path}: {e}")
            continue

        for enc in encs:
            X_list.append(enc.astype(np.float32))
            y_list.append(person)
            n_faces += 1

    if not X_list:
        raise RuntimeError("No face encodings were produced. Check your dataset quality.")

    X = np.vstack(X_list)
    y = np.array(y_list)

    enc_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(enc_path), X=X, y=y)

    return X, y, len(all_paths), n_faces


# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face Recognition MVP (register, encode, train, and recognize)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # register (stub for now)
    p_reg = sub.add_parser("register", help="Capture face crops via webcam (stub)")
    p_reg.add_argument("--name", required=True, help="Person name")
    p_reg.add_argument("--num", type=int, default=20, help="Number of samples to capture")

    # encode
    p_enc = sub.add_parser("encode", help="Encode dataset images into embeddings")
    p_enc.add_argument("--model", choices=["hog", "cnn"], default="hog", help="Face detector model")
    p_enc.add_argument("--upsample", type=int, default=1, help="Number of times to upsample")
    p_enc.add_argument(
        "--out", type=str, default=str(ENC_PATH), help="Output npz path (X,y)"
    )

    # train (stub for now)
    sub.add_parser("train", help="Train KNN on embeddings (stub)")

    # webcam (stub for now)
    sub.add_parser("webcam", help="Live webcam recognition (stub)")

    # recognize (stub for now)
    p_rec = sub.add_parser("recognize", help="Recognize faces in an image (stub)")
    p_rec.add_argument("--image", required=True, help="Path to image")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "register":
        print("register: not implemented yet. Use manual dataset for now.")
        return

    if args.cmd == "encode":
        out_path = Path(args.out)
        X, y, n_imgs, n_faces = encode_dataset(
            dataset_dir=DATASET_DIR,
            enc_path=out_path,
            model=args.model,
            upsample=args.upsample,
        )
        print(f"Saved encodings to {out_path}")
        print(f"Images scanned: {n_imgs}, faces encoded: {n_faces}, classes: {len(np.unique(y))}")
        return

    if args.cmd == "train":
        print("train: not implemented yet.")
        return

    if args.cmd == "webcam":
        print("webcam: not implemented yet.")
        return

    if args.cmd == "recognize":
        print("recognize: not implemented yet.")
        return


if __name__ == "__main__":
    main()
