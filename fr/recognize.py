from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from joblib import load
from sklearn.neighbors import KNeighborsClassifier
import face_recognition

from .paths import KNN_PATH
from .utils import draw_box_with_label, distance_to_confidence


def load_knn(knn_path: Path = KNN_PATH) -> KNeighborsClassifier:
    if not knn_path.exists():
        raise FileNotFoundError(f"KNN model not found: {knn_path}. Run 'train' first.")
    model: KNeighborsClassifier = load(str(knn_path))
    return model


def recognize_image(
    image_path: Path,
    knn_path: Path = KNN_PATH,
    detector_model: str = "hog",
    upsample: int = 1,
    threshold: float = 0.6,
    show: bool = False,
    out_path: Path | None = None,
) -> List[Dict]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    knn = load_knn(knn_path)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(
        rgb, number_of_times_to_upsample=upsample, model=detector_model
    )
    if not boxes:
        print("No faces found.")
        if out_path is not None:
            cv2.imwrite(str(out_path), bgr)
        if show:
            cv2.imshow("recognize", bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return []

    encs = face_recognition.face_encodings(rgb, boxes, num_jitters=1)
    preds = knn.predict(encs)
    dists, _ = knn.kneighbors(encs, n_neighbors=1, return_distance=True)
    dists = dists.reshape(-1)

    results = []
    for box, pred_label, dist in zip(boxes, preds, dists):
        conf = distance_to_confidence(float(dist), threshold)
        label = pred_label if dist <= threshold else "Unknown"
        results.append(
            {
                "box": box,
                "label": str(label),
                "distance": float(dist),
                "confidence": float(conf),
            }
        )

    for r in results:
        display_label = f"{r['label']} {r['confidence']:.2f}"
        draw_box_with_label(bgr, r["box"], display_label)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), bgr)
    if show:
        cv2.imshow("recognize", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results

