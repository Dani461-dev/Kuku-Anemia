#!/usr/bin/env python3
"""Sanity tests untuk core/masking.py (kmeans/otsu/grabcut + auto)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.masking import (  # noqa: E402
    best_nail_mask,
    nail_mask_grabcut,
    nail_mask_kmeans,
    nail_mask_otsu,
)


def synth_nail_crop(size: int = 60) -> np.ndarray:
    """Latar pink (kulit) + persegi lebih terang (kuku) di tengah."""
    img = np.full((size, size, 3), (200, 180, 170), np.uint8)  # BGR-ish kulit
    m = size // 2 - 14
    img[m:m + 28, m:m + 28] = (235, 220, 210)  # "kuku" lebih terang/saturasi rendah
    return img


def main() -> int:
    crop = synth_nail_crop()
    h, w = crop.shape[:2]

    for name, fn in [("kmeans", nail_mask_kmeans), ("otsu", nail_mask_otsu),
                     ("grabcut", nail_mask_grabcut)]:
        mask = fn(crop)
        assert mask.shape == (h, w), f"{name}: bentuk mask {mask.shape}"
        assert set(np.unique(mask)) <= {0, 255}, f"{name}: mask harus biner 0/255"
        frac = float(np.mean(mask > 0))
        print(f"  {name}: coverage={frac:.3f}")
        assert 0.0 < frac < 1.0, f"{name}: coverage tidak wajar"

    auto = best_nail_mask(crop, method="auto")
    assert auto is not None, "auto mask harus menghasilkan mask pada crop sintetis"
    cov = float(np.mean(auto > 0))
    print(f"  auto: coverage={cov:.3f}")
    assert 0.15 < cov < 0.99, "auto: coverage di luar batas kewajaran"

    assert best_nail_mask(np.zeros((5, 5, 3), np.uint8), method="auto") is None, "crop terlalu kecil -> None"

    print("OK: masking (kmeans/otsu/grabcut/auto) lolos")
    return 0


if __name__ == "__main__":
    sys.exit(main())