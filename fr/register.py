from __future__ import annotations

from pathlib import Path

import cv2
import face_recognition

from .paths import DATASET_DIR
from .utils import draw_box_with_label, expand_and_clip_box


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
            small = (
                cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame
            )
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            boxes_small = face_recognition.face_locations(
                rgb_small, number_of_times_to_upsample=upsample, model=detector_model
            )

            target_box = None
            if boxes_small:
                inv = (1.0 / scale) if scale != 0 else 1.0
                def area(b):
                    t, r, btm, l = b
                    return (r - l) * (btm - t)
                b = max(boxes_small, key=area)
                t, r, btm, l = b
                t = int(round(t * inv)); r = int(round(r * inv)); btm = int(round(btm * inv)); l = int(round(l * inv))
                target_box = (t, r, btm, l)
                target_box = expand_and_clip_box(target_box, display.shape[1], display.shape[0], margin)
                draw_box_with_label(display, target_box, f"{name} {saved+1}/{num}")

            cv2.putText(
                display,
                "Press SPACE to capture, Q to quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("register", display)
            key = cv2.waitKey(1) & 0xFF

            should_capture = False
            if key == ord("q"):
                break
            if key == ord(" "):
                should_capture = True
            if target_box is not None and key == 255:
                should_capture = True

            if should_capture and target_box is not None:
                t, r, btm, l = target_box
                crop = frame[t:btm, l:r]
                if crop.size == 0:
                    continue
                h, w = crop.shape[:2]
                if h < 40 or w < 40:
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

