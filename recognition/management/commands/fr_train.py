from __future__ import annotations

from django.core.management.base import BaseCommand
from pathlib import Path
import numpy as np

from fr.paths import ENC_PATH, KNN_PATH
from fr.train import train_knn


class Command(BaseCommand):
    help = "Train a KNN classifier from encodings and save to models/knn.joblib"

    def add_arguments(self, parser):
        parser.add_argument("--enc", type=str, default=str(ENC_PATH))
        parser.add_argument("--out", type=str, default=str(KNN_PATH))
        parser.add_argument("--k", type=int, default=0)
        parser.add_argument("--weights", choices=["uniform", "distance"], default="distance")
        parser.add_argument("--algorithm", choices=["auto", "ball_tree", "kd_tree", "brute"], default="auto")
        parser.add_argument("--metric", choices=["euclidean", "minkowski", "cosine"], default="euclidean")

    def handle(self, *args, **opts):
        enc_path = Path(opts["enc"]) if opts.get("enc") else ENC_PATH
        out_path = Path(opts["out"]) if opts.get("out") else KNN_PATH
        n_neighbors = None if int(opts["k"]) == 0 else int(opts["k"])
        knn, k_used, classes = train_knn(
            enc_path=enc_path,
            knn_path=out_path,
            n_neighbors=n_neighbors,
            weights=opts["weights"],
            algorithm=opts["algorithm"],
            metric=opts["metric"],
        )
        self.stdout.write(self.style.SUCCESS(
            f"Saved {out_path} | k={k_used} classes={len(classes)}"
        ))

