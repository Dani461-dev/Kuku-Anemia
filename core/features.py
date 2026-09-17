"""Feature extraction ported from the original notebook.

Reference behaviour:
- cut_image(img, low=0.2, high=0.8) crops the inner 60% of a region.
- calculate_features() computes intensity percentiles per RGB channel.
- Features are later divided, per colour channel, by the median of a white
  reference region (illumination normalisation).

Here we additionally support a binary mask (e.g. K-means/Otsu nail mask):
percentiles are then computed only over masked pixels.
"""

from __future__ import annotations

import numpy as np

from core.config import COLORS, PERCENTILE_LEVELS


def cut_image(img: np.ndarray, low: float = 0.2, high: float = 0.8) -> np.ndarray:
    """Crop the inner (high-low) fraction of both dimensions.

    Works on 2D (single channel) or 3D (multi channel) arrays.
    """
    h, w = img.shape[:2]
    return img[int(low * h) : int(high * h), int(low * w) : int(high * w)]


def calculate_features(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    percentile_levels: list[int] | None = None,
    low: float = 0.2,
    high: float = 0.8,
) -> dict[str, float]:
    """Compute RGB intensity percentiles over a region.

    If mask is given (boolean, same HxW as img) the percentiles are computed
    over the masked pixels only. Otherwise the inner 60% crop is used (exactly
    like the original notebook).
    """
    percentile_levels = percentile_levels or PERCENTILE_LEVELS
    if mask is not None:
        mask = mask.astype(bool)
        if mask.shape[:2] != img.shape[:2]:
            raise ValueError("mask shape does not match image")
        valid = mask
    else:
        valid = None

    features: dict[str, float] = {}
    for chan_id, color in enumerate(COLORS):
        channel = img[:, :, chan_id]
        if valid is not None:
            pixels = channel[valid]
        else:
            cut = cut_image(channel, low=low, high=high)
            pixels = cut.ravel()
        if pixels.size == 0:
            raise ValueError("no valid pixels in region")
        for level in percentile_levels:
            features[f"{color}_p={level}"] = float(np.percentile(pixels, level))
    return features


def normalize_by_white(features: dict[str, float], white_median: dict[str, float]) -> dict[str, float]:
    """Normalize percentile features by the white median of their colour channel.

    Naming convention: tissue-prefixed (NAIL_/SKIN_) percentiles are divided by
    the matching white channel. Everything else (e.g. EXT_* extended features,
    already computed on white-balanced pixels) is passed through unchanged.
    """
    out: dict[str, float] = {}
    for name, value in features.items():
        parts = name.split("_")
        if parts[0] in ("NAIL", "SKIN"):
            channel = parts[1]
            denom = white_median[channel]
            if denom <= 0:
                denom = 255.0
            out[name] = value / denom
        else:
            out[name] = value
    return out


def feature_names(prefix: str = "", levels: list[int] | None = None) -> list[str]:
    levels = levels or PERCENTILE_LEVELS
    names = []
    for color in COLORS:
        for level in levels:
            names.append(f"{prefix}{color}_p={level}")
    return names