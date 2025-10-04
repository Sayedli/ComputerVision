from __future__ import annotations

"""CLI wiring for the Face Recognition MVP.

Defines subcommands and forwards to implementation modules.
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np

from .paths import DATASET_DIR, ENC_PATH, KNN_PATH
from .encode import encode_dataset
from .train import train_knn
from .webcam import recognize_webcam
from .recognize import recognize_image
from .register import register_person


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face Recognition MVP (register, encode, train, recognize, webcam)"
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
    p_enc.add_argument("--out", type=str, default=str(ENC_PATH), help="Output npz path (X,y)")

    # train
    p_train = sub.add_parser("train", help="Train KNN on embeddings and save model")
    p_train.add_argument("--enc", type=str, default=str(ENC_PATH), help="Input encodings npz path")
    p_train.add_argument("--out", type=str, default=str(KNN_PATH), help="Output KNN joblib path")
    p_train.add_argument("--k", type=int, default=0, help="Neighbors (0 = auto sqrt(N))")
    p_train.add_argument("--weights", choices=["uniform", "distance"], default="distance", help="KNN weights")
    p_train.add_argument(
        "--algorithm", choices=["auto", "ball_tree", "kd_tree", "brute"], default="auto", help="KNN algorithm"
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
