from __future__ import annotations

import numpy as np

from fr.utils import distance_to_confidence, expand_and_clip_box


def test_distance_to_confidence_behaviour():
    thr = 0.6
    assert distance_to_confidence(0.0, thr) == 1.0
    assert distance_to_confidence(thr, thr) == 0.0
    # Above threshold should clamp to 0
    assert distance_to_confidence(1.2, thr) == 0.0
    # Monotonic decreasing
    c1 = distance_to_confidence(0.1, thr)
    c2 = distance_to_confidence(0.2, thr)
    assert c1 > c2


def test_expand_and_clip_box():
    # box near the top-left corner, margin should be clipped to image bounds
    box = (5, 25, 25, 5)  # (top, right, bottom, left)
    w, h = 30, 30
    t, r, b, l = expand_and_clip_box(box, w, h, margin=0.5)
    # All coords should be within image bounds
    assert 0 <= l <= r < w
    assert 0 <= t <= b < h
    # Expanded box should not be smaller than original
    assert (r - l) >= (25 - 5)
    assert (b - t) >= (25 - 5)

