"""Feature pipeline for one photo / one patient.

Stages (given a set of detected nail & skin boxes):
  1. white reference (fixed | chart ArUco | auto)
  2. masking (K-means/Otsu) inside the nail box (optional)
  3. percentile features (RGB) on nail + skin
  4. normalize features by the white median (per channel)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config as cfg
from .calibration import ChartDetector
from .detectors import DetectedFinger, GTDetector
from .extended import extended_features, whiten_pixels
from .features import calculate_features, feature_names, normalize_by_white
from .masking import best_nail_mask
from .util import clip_box


def load_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def white_fixed(img_rgb: np.ndarray) -> dict[str, float]:
    region = img_rgb[cfg.WHITE_ROWS[0]:cfg.WHITE_ROWS[1], cfg.WHITE_COLS[0]:cfg.WHITE_COLS[1]]
    if region.size == 0:
        return {"R": 255.0, "G": 255.0, "B": 255.0}
    med = np.median(region.reshape(-1, 3), axis=0)
    return {"R": float(med[0]), "G": float(med[1]), "B": float(med[2])}


def white_auto(img_rgb: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([255, 35, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        med = np.median(img_rgb.reshape(-1, 3), axis=0)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        pixels = img_rgb[labels == best]
        med = np.median(pixels, axis=0) if len(pixels) else np.median(img_rgb.reshape(-1, 3), axis=0)
    return {"R": float(med[0]), "G": float(med[1]), "B": float(med[2])}


def white_median_for(img_rgb: np.ndarray, source: str = "fixed",
                     chart_detector: ChartDetector | None = None) -> dict[str, float]:
    if source == "chart":
        if chart_detector is None:
            chart_detector = ChartDetector()
        wm = chart_detector.white_median(img_rgb)
        if wm:
            return wm
        return white_auto(img_rgb)
    if source == "auto":
        return white_auto(img_rgb)
    return white_fixed(img_rgb)


def crop_from_box(img_rgb: np.ndarray, box) -> np.ndarray:
    t, l, b, r = clip_box(box, *img_rgb.shape[:2])
    return img_rgb[t:b, l:r].copy()


def _mask_for(crop: np.ndarray, use_mask: bool, mask_method: str) -> tuple[np.ndarray | None, float]:
    if not use_mask:
        return None, 1.0
    m = best_nail_mask(crop, method=mask_method)
    frac = float(np.mean(m > 0)) if m is not None else 1.0
    return m, frac


def _region_features(img_rgb: np.ndarray, box, use_mask: bool, mask_method: str = "auto") -> tuple[dict[str, float], float, np.ndarray | None]:
    crop = crop_from_box(img_rgb, box)
    if crop.size == 0:
        return {f: 0.0 for f in feature_names()}, 0.0, None
    m, frac = _mask_for(crop, use_mask, mask_method)
    return calculate_features(crop, mask=m), frac, m


def select_middle_finger(detected: list[DetectedFinger]) -> DetectedFinger | None:
    """Pick the finger corresponding to the dataset's NAIL_2/SKIN_2 (middle one)."""
    if not detected:
        return None
    # order by box vertical (y) centre, the middle of the sorted list is finger #2
    ordered = sorted(detected, key=lambda d: (d.nail_box[0] + d.nail_box[2]) / 2)
    return ordered[len(ordered) // 2]


def patient_features(
    img_rgb: np.ndarray,
    boxes: DetectedFinger,
    white_source: str = "fixed",
    use_mask: bool = True,
    mask_method: str = "auto",
    extended: bool = False,
    chart_detector: ChartDetector | None = None,
) -> dict:
    """Raw normalized features (42) + optional extended + metadata for one finger."""
    wm = white_median_for(img_rgb, white_source, chart_detector)
    nail_feats, mask_frac, nail_mask = _region_features(img_rgb, boxes.nail_box, use_mask, mask_method)
    skin_feats, _, _ = _region_features(img_rgb, boxes.skin_box, use_mask=False)
    raw = {f"NAIL_{k}": v for k, v in nail_feats.items()}
    raw.update({f"SKIN_{k}": v for k, v in skin_feats.items()})
    if extended:
        nail_crop = crop_from_box(img_rgb, boxes.nail_box)
        skin_crop = crop_from_box(img_rgb, boxes.skin_box)
        nail_w = whiten_pixels(nail_crop, wm)
        skin_w = whiten_pixels(skin_crop, wm)
        raw.update(extended_features(nail_w, wm, nail_mask, skin_w))
    norm = normalize_by_white(raw, wm)
    norm["_PATIENT_ID"] = 0
    norm["_MASK_COVERAGE"] = mask_frac
    return norm