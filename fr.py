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
# Train KNN on encodings
# ----------------------------
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
    """
    Train a KNN classifier on saved encodings and persist it to disk.

    Returns: (knn, k_used, classes)
    """
    X, y = load_encodings(enc_path)
    n_samples = len(X)
    if n_samples == 0:
        raise RuntimeError("Encodings are empty; nothing to train.")

    # Choose a default k ~ sqrt(N) if not provided
    if n_neighbors is None or n_neighbors <= 0:
        n_neighbors = max(1, int(round(np.sqrt(n_samples))))
    # Ensure k does not exceed number of samples
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


# ----------------------------
# Recognize faces in an image
# ----------------------------
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
) -> list[dict]:
    """
    Recognize faces in a single image using the trained KNN classifier.

    Returns a list of result dicts with keys: box, label, distance, confidence.
    Optionally draws boxes/labels and shows/saves the annotated image.
    """
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
        # Optionally save/show the original image anyway
        if out_path is not None:
            cv2.imwrite(str(out_path), bgr)
        if show:
            cv2.imshow("recognize", bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return []

    encs = face_recognition.face_encodings(rgb, boxes, num_jitters=1)
    # Predict labels
    preds = knn.predict(encs)
    # Get nearest neighbor distance per face
    dists, _ = knn.kneighbors(encs, n_neighbors=1, return_distance=True)
    dists = dists.reshape(-1)

    results = []
    for box, pred_label, dist in zip(boxes, preds, dists):
        conf = distance_to_confidence(float(dist), threshold)
        label = pred_label if dist <= threshold else "Unknown"
        results.append({
            "box": box,
            "label": str(label),
            "distance": float(dist),
            "confidence": float(conf),
        })

    # Draw and optionally show/save
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


# ----------------------------
# Live webcam recognition
# ----------------------------
def recognize_webcam(
    knn_path: Path = KNN_PATH,
    detector_model: str = "hog",
    upsample: int = 0,
    threshold: float = 0.6,
    camera: int = 0,
    scale: float = 0.25,
    process_every: int = 2,
) -> None:
    """
    Run a webcam loop that detects faces, predicts labels with a trained KNN,
    and displays annotated frames. Press 'q' to quit.
    """
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

            # Optionally process every Nth frame for speed
            do_process = (frame_idx % max(1, process_every) == 0)
            frame_idx += 1

            display = frame.copy()
            results = []
            if do_process:
                # Resize to speed up detection/encoding
                if scale != 1.0:
                    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                else:
                    small = frame
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
                            # scale back box to original frame size
                            top, right, bottom, left = box_s
                            top = int(round(top * inv))
                            right = int(round(right * inv))
                            bottom = int(round(bottom * inv))
                            left = int(round(left * inv))
                            conf = distance_to_confidence(float(dist), threshold)
                            label = pred_label if dist <= threshold else "Unknown"
                            results.append({
                                "box": (top, right, bottom, left),
                                "label": str(label),
                                "distance": float(dist),
                                "confidence": float(conf),
                            })

            # Draw last computed results
            for r in results:
                display_label = f"{r['label']} {r['confidence']:.2f}"
                draw_box_with_label(display, r["box"], display_label)

            cv2.imshow("webcam", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ----------------------------
# Register: collect samples via webcam
# ----------------------------
def _expand_and_clip_box(box: Tuple[int, int, int, int], w: int, h: int, margin: float = 0.2) -> Tuple[int, int, int, int]:
    top, right, bottom, left = box
    bw = right - left
    bh = bottom - top
    mt = int(round(bh * margin))
    ml = int(round(bw * margin))
    top = max(0, top - mt)
    bottom = min(h - 1, bottom + mt)
    left = max(0, left - ml)
    right = min(w - 1, right + ml)
    return top, right, bottom, left


def register_person(
    name: str,
    num: int = 20,
    camera: int = 0,
    detector_model: str = "hog",
    upsample: int = 0,
    scale: float = 0.5,
    margin: float = 0.2,
    mirror: bool = True,
) -> int:
    """
    Capture `num` face crops for a person from the webcam and save under dataset/<name>/.
    Returns the number of saved images.
    """
    out_dir = DATASET_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera}")

    saved = 0
    idx = 0
    try:
        while saved < num:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] Failed to read frame from camera.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            display = frame.copy()
            # Resize for detection speed
            if scale != 1.0:
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            else:
                small = frame
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            boxes_small = face_recognition.face_locations(
                rgb_small, number_of_times_to_upsample=upsample, model=detector_model
            )

            target_box = None
            if boxes_small:
                # choose the largest face
                inv = (1.0 / scale) if scale != 0 else 1.0
                def area(b):
                    t, r, btm, l = b
                    return (r - l) * (btm - t)
                b = max(boxes_small, key=area)
                t, r, btm, l = b
                t = int(round(t * inv)); r = int(round(r * inv)); btm = int(round(btm * inv)); l = int(round(l * inv))
                target_box = (t, r, btm, l)
                target_box = _expand_and_clip_box(target_box, display.shape[1], display.shape[0], margin)
                draw_box_with_label(display, target_box, f"{name} {saved+1}/{num}")

            # HUD text
            cv2.putText(display, "Press SPACE to capture, Q to quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("register", display)
            key = cv2.waitKey(1) & 0xFF

            should_capture = False
            if key == ord('q'):
                break
            if key == ord(' '):
                should_capture = True
            # Auto-capture if a face is present and not pressing keys
            if target_box is not None and key == 255:
                should_capture = True

            if should_capture and target_box is not None:
                t, r, btm, l = target_box
                crop = frame[t:btm, l:r]
                if crop.size == 0:
                    continue
                # simple normalization: ensure minimum size
                h, w = crop.shape[:2]
                if h < 40 or w < 40:
                    # skip tiny crops
                    continue
                fname = out_dir / f"{name}_{idx:04d}.jpg"
                idx += 1
                ok = cv2.imwrite(str(fname), crop)
                if ok:
                    saved += 1
                else:
                    print(f"[warn] Failed to save {fname}")

        return saved
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face Recognition MVP (register, encode, train, and recognize)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # register
    p_reg = sub.add_parser("register", help="Capture face crops via webcam to dataset/<name>")
    p_reg.add_argument("--name", required=True, help="Person name (folder under dataset/)")
    p_reg.add_argument("--num", type=int, default=20, help="Number of samples to capture")
    p_reg.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p_reg.add_argument("--model", choices=["hog", "cnn"], default="hog", help="Face detector model")
    p_reg.add_argument("--upsample", type=int, default=0, help="Number of times to upsample (detector)")
    p_reg.add_argument("--scale", type=float, default=0.5, help="Downscale factor for detection (0.5 = half)")
    p_reg.add_argument("--margin", type=float, default=0.2, help="Extra margin around the face crop (fraction)")
    p_reg.add_argument("--no-mirror", action="store_true", help="Disable mirroring for preview")

    # encode
    p_enc = sub.add_parser("encode", help="Encode dataset images into embeddings")
    p_enc.add_argument("--model", choices=["hog", "cnn"], default="hog", help="Face detector model")
    p_enc.add_argument("--upsample", type=int, default=1, help="Number of times to upsample")
    p_enc.add_argument(
        "--out", type=str, default=str(ENC_PATH), help="Output npz path (X,y)"
    )

    # train
    p_train = sub.add_parser("train", help="Train KNN on embeddings and save model")
    p_train.add_argument("--enc", type=str, default=str(ENC_PATH), help="Input encodings npz path")
    p_train.add_argument("--out", type=str, default=str(KNN_PATH), help="Output KNN joblib path")
    p_train.add_argument("--k", type=int, default=0, help="Neighbors (0 = auto sqrt(N))")
    p_train.add_argument(
        "--weights", choices=["uniform", "distance"], default="distance", help="KNN weights"
    )
    p_train.add_argument(
        "--algorithm",
        choices=["auto", "ball_tree", "kd_tree", "brute"],
        default="auto",
        help="KNN algorithm",
    )
    p_train.add_argument(
        "--metric", choices=["euclidean", "minkowski", "cosine"], default="euclidean", help="Distance metric"
    )

    # webcam
    p_cam = sub.add_parser("webcam", help="Live webcam recognition with KNN labels")
    p_cam.add_argument("--knn", type=str, default=str(KNN_PATH), help="Path to trained KNN joblib")
    p_cam.add_argument("--model", choices=["hog", "cnn"], default="hog", help="Face detector model")
    p_cam.add_argument("--upsample", type=int, default=0, help="Number of times to upsample (detector)")
    p_cam.add_argument("--threshold", type=float, default=0.6, help="Distance threshold for 'Unknown'")
    p_cam.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p_cam.add_argument("--scale", type=float, default=0.25, help="Downscale factor for processing (0.25 = quarter)")
    p_cam.add_argument("--process-every", type=int, default=2, help="Process every N frames for speed")

    # recognize
    p_rec = sub.add_parser("recognize", help="Recognize faces in a single image")
    p_rec.add_argument("--image", required=True, help="Path to image")
    p_rec.add_argument("--model", choices=["hog", "cnn"], default="hog", help="Face detector model")
    p_rec.add_argument("--upsample", type=int, default=1, help="Number of times to upsample")
    p_rec.add_argument("--threshold", type=float, default=0.6, help="Distance threshold for 'Unknown'")
    p_rec.add_argument("--knn", type=str, default=str(KNN_PATH), help="Path to trained KNN joblib")
    p_rec.add_argument("--out", type=str, default="", help="Optional output image path to save annotations")
    p_rec.add_argument("--show", action="store_true", help="Show annotated image in a window")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "register":
        saved = register_person(
            name=args.name,
            num=args.num,
            camera=args.camera,
            detector_model=args.model,
            upsample=args.upsample,
            scale=args.scale,
            margin=args.margin,
            mirror=(not args.no_mirror),
        )
        print(f"Saved {saved} images to {DATASET_DIR / args.name}")
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
        enc_path = Path(args.enc)
        out_path = Path(args.out)
        knn, k_used, classes = train_knn(
            enc_path=enc_path,
            knn_path=out_path,
            n_neighbors=(None if args.k == 0 else args.k),
            weights=args.weights,
            algorithm=args.algorithm,
            metric=args.metric,
        )
        print(f"Saved KNN to {out_path} (k={k_used}, classes={len(classes)})")
        return

    if args.cmd == "webcam":
        recognize_webcam(
            knn_path=Path(args.knn),
            detector_model=args.model,
            upsample=args.upsample,
            threshold=args.threshold,
            camera=args.camera,
            scale=args.scale,
            process_every=args.process_every,
        )
        return

    if args.cmd == "recognize":
        img_path = Path(args.image)
        knn_path = Path(args.knn)
        out_path = Path(args.out) if args.out else None
        results = recognize_image(
            image_path=img_path,
            knn_path=knn_path,
            detector_model=args.model,
            upsample=args.upsample,
            threshold=args.threshold,
            show=args.show,
            out_path=out_path,
        )
        for r in results:
            t, rgt, btm, lft = r["box"]
            print(
                f"box=({lft},{t},{rgt},{btm}) label={r['label']} dist={r['distance']:.3f} conf={r['confidence']:.2f}"
            )
        if out_path is not None:
            print(f"Saved annotated image to {out_path}")
        return


if __name__ == "__main__":
    main()
