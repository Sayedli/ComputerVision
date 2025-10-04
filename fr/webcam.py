from __future__ import annotations

"""Live webcam recognition loop using a trained KNN model."""

from pathlib import Path

import cv2
import face_recognition

from .paths import KNN_PATH
from .recognize import load_knn
from .utils import draw_box_with_label, distance_to_confidence


def recognize_webcam(
    knn_path: Path = KNN_PATH,
    detector_model: str = "hog",
    upsample: int = 0,
    threshold: float = 0.6,
    camera: int = 0,
    scale: float = 0.25,
    process_every: int = 2,
) -> None:
    knn = load_knn(knn_path)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] Failed to read frame from camera.")
                break

            do_process = frame_idx % max(1, process_every) == 0
            frame_idx += 1

            display = frame.copy()
            results = []
            if do_process:
                small = (
                    cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame
                )
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                boxes_small = face_recognition.face_locations(
                    rgb_small, number_of_times_to_upsample=upsample, model=detector_model
                )
                if boxes_small:
                    encs = face_recognition.face_encodings(rgb_small, boxes_small, num_jitters=1)
                    if len(encs) > 0:
                        preds = knn.predict(encs)
                        dists, _ = knn.kneighbors(encs, n_neighbors=1, return_distance=True)
                        dists = dists.reshape(-1)

                        inv = (1.0 / scale) if scale != 0 else 1.0
                        for box_s, pred_label, dist in zip(boxes_small, preds, dists):
                            top, right, bottom, left = box_s
                            top = int(round(top * inv))
                            right = int(round(right * inv))
                            bottom = int(round(bottom * inv))
                            left = int(round(left * inv))
                            conf = distance_to_confidence(float(dist), threshold)
                            label = pred_label if dist <= threshold else "Unknown"
                            results.append(
                                {
                                    "box": (top, right, bottom, left),
                                    "label": str(label),
                                    "distance": float(dist),
                                    "confidence": float(conf),
                                }
                            )

            for r in results:
                display_label = f"{r['label']} {r['confidence']:.2f}"
                draw_box_with_label(display, r["box"], display_label)

            cv2.imshow("webcam", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
