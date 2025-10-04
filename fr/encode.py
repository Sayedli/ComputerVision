from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import face_recognition

from .paths import DATASET_DIR, ENC_PATH
from .utils import list_persons, image_paths_for_person


def encode_dataset(
    dataset_dir: Path = DATASET_DIR,
    enc_path: Path = ENC_PATH,
    model: str = "hog",
    upsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
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

