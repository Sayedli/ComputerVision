from __future__ import annotations

from django.core.management.base import BaseCommand
from pathlib import Path
import numpy as np

from fr.paths import DATASET_DIR, ENC_PATH
from fr.encode import encode_dataset


class Command(BaseCommand):
    help = "Encode faces from dataset/ into embeddings and save to models/encodings.npz"

    def add_arguments(self, parser):
        parser.add_argument("--model", choices=["hog", "cnn"], default="hog")
        parser.add_argument("--upsample", type=int, default=1)
        parser.add_argument("--out", type=str, default=str(ENC_PATH))

    def handle(self, *args, **opts):
        model = opts["model"]
        upsample = opts["upsample"]
        out_path = Path(opts["out"]) if opts.get("out") else ENC_PATH
        X, y, n_imgs, n_faces = encode_dataset(
            dataset_dir=DATASET_DIR,
            enc_path=out_path,
            model=model,
            upsample=upsample,
        )
        classes = np.unique(y)
        self.stdout.write(self.style.SUCCESS(
            f"Saved {out_path} | images={n_imgs} faces={n_faces} classes={len(classes)}"
        ))

