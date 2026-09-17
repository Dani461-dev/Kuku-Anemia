"""Masking of the nail region inside a tight bounding-box crop.

Three strategies are provided:
1. K-means clustering (k=2) in RGB space; the cluster containing the crop
   center pixel is treated as the nail.
2. Otsu thresholding on the HSV Value channel; the class containing the crop
   center pixel is used.
3. GrabCut with an inset rectangle, refined by a low-saturation filter
   (nail is typically less saturated than skin) — per NAIL_ANALYSIS_PIPELINE.md
   §2.1 recommendation.

All are followed by morphological cleanup (open + close + keep largest blob).
"""

from __future__ import annotations

import cv2
import numpy as np


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask
    # keep the largest non-background component
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    return (labels == largest).astype(np.uint8) * 255


def nail_mask_kmeans(img_rgb: np.ndarray, k: int = 2, seed: int = 42) -> np.ndarray:
    """Return a binary mask (HxW, 0-255) of the nail cluster for the crop."""
    h, w = img_rgb.shape[:2]
    flat = img_rgb.reshape(-1, 3).astype(np.float32)
    if len(flat) < max(k * 8, 64):
        return np.full((h, w), 255, np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(flat, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    center_idx = min(h // 2 * w + w // 2, len(labels) - 1)
    center_cluster = int(labels[center_idx])
    mask_flat = labels.reshape(h, w) == center_cluster
    return _clean_mask(mask_flat.astype(np.uint8) * 255)


def nail_mask_otsu(img_rgb: np.ndarray) -> np.ndarray:
    """Otsu threshold on the Value channel; keep the class of the center pixel."""
    h, w = img_rgb.shape[:2]
    v = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    _, th = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    center_val = th[h // 2, w // 2] > 0
    mask = (th > 0) if center_val else (th == 0)
    return _clean_mask(mask.astype(np.uint8))


def nail_mask_grabcut(img_rgb: np.ndarray, inset: int = 5) -> np.ndarray:
    """GrabCut segmentation refined by a low-saturation filter.

    The crop is expected to be a tight nail box, so an inset rectangle seeds
    the initial foreground. Pixels with low saturation (whitish nail / specular
    highlights) that GrabCut drops are re-added via a saturation band filter,
    mirroring the original notebook's `mask_nail_region` behaviour.
    """
    h, w = img_rgb.shape[:2]
    if h < 12 or w < 12:
        return np.full((h, w), 255, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    ins = min(inset, h // 4, w // 4)
    rect = (ins, ins, max(1, w - 2 * ins), max(1, h - 2 * ins))
    try:
        cv2.grabCut(img_rgb, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        gc = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    except cv2.error:
        gc = np.ones((h, w), np.uint8)
    s = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    sat_ok = ((s > 20) & (s < 200)).astype(np.uint8)
    return _clean_mask((gc & sat_ok).astype(np.uint8) * 255)


def best_nail_mask(img_rgb: np.ndarray, min_coverage: float = 0.15, max_coverage: float = 0.99, method: str = "auto") -> np.ndarray | None:
    """Choose the best nail mask. method: auto | kmeans | otsu | grabcut.

    auto tries K-means then Otsu and keeps the first candidate with a sane
    coverage. GrabCut is offered explicitly (it needs a larger crop than the
    small GT boxes in this dataset, but can help at inference on full photos).
    Returns None if nothing is acceptable (caller should fall back to the plain
    inner-60% crop). mask pixels: >0 = nail.
    """
    h, w = img_rgb.shape[:2]
    if h * w < 64 or min(h, w) < 3:
        return None
    if method == "kmeans":
        candidates = [nail_mask_kmeans(img_rgb)]
    elif method == "otsu":
        candidates = [nail_mask_otsu(img_rgb)]
    elif method == "grabcut":
        candidates = [nail_mask_grabcut(img_rgb)]
    else:
        candidates = [nail_mask_kmeans(img_rgb), nail_mask_otsu(img_rgb)]
    for candidate in candidates:
        frac = float(np.mean(candidate > 0))
        if min_coverage < frac < max_coverage:
            return candidate
    return None