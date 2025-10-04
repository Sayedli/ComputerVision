from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def list_persons(dataset_dir: Path) -> List[str]:
    if not dataset_dir.exists():
        return []
    return sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])


def image_paths_for_person(dataset_dir: Path, person: str) -> List[Path]:
    pdir = dataset_dir / person
    if not pdir.exists():
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in pdir.rglob("*") if p.suffix.lower() in exts])


def draw_box_with_label(frame: np.ndarray, box: Tuple[int, int, int, int], label: str) -> None:
    top, right, bottom, left = box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 10 if top - 10 > 10 else top + 20
    cv2.rectangle(frame, (left, y - 20), (left + 6 * len(label), y), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, label, (left + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def distance_to_confidence(dist: float, threshold: float = 0.6) -> float:
    conf = max(0.0, 1.0 - (dist / threshold))
    return float(min(1.0, conf))


def expand_and_clip_box(box: Tuple[int, int, int, int], w: int, h: int, margin: float = 0.2) -> Tuple[int, int, int, int]:
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

