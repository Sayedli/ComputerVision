#!/usr/bin/env python3
from __future__ import annotations

"""Copy sample images from samples/<person>/ into dataset/<person>/.

Skips non-image files. Creates destination folders as needed.
"""

from pathlib import Path
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = root / "samples"
    dataset = root / "dataset"
    if not samples.exists():
        print(f"No samples directory found at {samples}")
        return
    copied = 0
    for person_dir in sorted(p for p in samples.iterdir() if p.is_dir()):
        out_dir = dataset / person_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in person_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                dest = out_dir / p.name
                shutil.copy2(p, dest)
                copied += 1
    print(f"Copied {copied} files into {dataset}")


if __name__ == "__main__":
    main()

