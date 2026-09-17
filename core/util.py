"""Small box/image helpers."""

from __future__ import annotations

import numpy as np


def iou(a, b) -> float:
    """Intersection over union of [top, left, bottom, right] pixel boxes."""
    ta, la, ba, ra = a
    tb, lb, bb, rb = b
    iw = max(0, min(ra, rb) - max(la, lb))
    ih = max(0, min(ba, bb) - max(ta, tb))
    inter = iw * ih
    area_a = max(1, (ba - ta)) * max(1, (ra - la))
    area_b = max(1, (bb - tb)) * max(1, (rb - lb))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clip_box(box, img_h: int, img_w: int) -> list[int]:
    top, left, bottom, right = box
    return [
        max(0, min(top, img_h - 1)),
        max(0, min(left, img_w - 1)),
        max(1, min(bottom, img_h)),
        max(1, min(right, img_w)),
    ]