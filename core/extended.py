"""Extended physiological features (HSV / LAB / chromaticity / nail-skin contrast).

These complement the 42 percentile-RGB features. They are computed on crops
after pixel-level white-balancing (divide each channel by the same white median
used for the percentile features), which keeps the whole feature vector on one
illumination-normalised basis.

Reference: the deprecated 74-feature scheme (models/legacy_74features/extract_features.py)
— kept here as a library for the accuracy experiment.
"""

from __future__ import annotations

import numpy as np
import cv2


def whiten_pixels(crop_rgb: np.ndarray, white_median: dict[str, float]) -> np.ndarray:
    """Per-channel white-balance: pixel / white_median * 255, clipped to 0..255."""
    img = crop_rgb.astype(np.float32)
    scale = np.array([white_median["R"], white_median["G"], white_median["B"]], dtype=np.float32)
    scale[scale <= 0] = 255.0
    img = img / scale * 255.0
    return np.clip(img, 0, 255)


def circular_mean_hue(hues: np.ndarray) -> float:
    """Circular mean of OpenCV hue (0..179)."""
    radians = hues * (2 * np.pi / 180.0)
    mean_angle = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return float(mean_angle * 180.0 / (2 * np.pi))


def region_stats(pixels_rgb_norm: np.ndarray, prefix: str) -> dict[str, float]:
    """HSV/LAB/chromaticity mean|median|std for one region (nail or skin).

    Input pixels are RGB (0..255, already white-balanced).
    """
    if len(pixels_rgb_norm) == 0:
        return {}
    px = pixels_rgb_norm.reshape(-1, 1, 3).astype(np.uint8)
    rgb = pixels_rgb_norm.astype(np.float32)
    hsv = cv2.cvtColor(px, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    out: dict[str, float] = {}
    pre = f"{prefix}_"

    rgb_sum = rgb.sum(axis=1, keepdims=True) + 1e-6
    chrom = rgb / rgb_sum
    for i, ch in enumerate(["r", "g", "b"]):
        out[f"{pre}chrom_{ch}_mean"] = float(np.mean(chrom[:, i]))
        out[f"{pre}chrom_{ch}_median"] = float(np.median(chrom[:, i]))
        out[f"{pre}chrom_{ch}_std"] = float(np.std(chrom[:, i]))

    for ch, data in [("h", hsv[:, 0]), ("s", hsv[:, 1]), ("v", hsv[:, 2])]:
        out[f"{pre}hsv_{ch}_mean"] = float(np.mean(data))
        out[f"{pre}hsv_{ch}_median"] = float(np.median(data))
        out[f"{pre}hsv_{ch}_std"] = float(np.std(data))
    out[f"{pre}hsv_h_circular_mean"] = circular_mean_hue(hsv[:, 0])

    for ch, data in [("l", lab[:, 0]), ("a", lab[:, 1]), ("b", lab[:, 2])]:
        out[f"{pre}lab_{ch}_mean"] = float(np.mean(data))
        out[f"{pre}lab_{ch}_median"] = float(np.median(data))
        out[f"{pre}lab_{ch}_std"] = float(np.std(data))
    return out


def contrast_features(nail_rgb: np.ndarray, skin_rgb: np.ndarray) -> dict[str, float]:
    """LAB / HSV / chromaticity nail-minus-skin diff & ratio on normalised RGB."""
    if len(nail_rgb) == 0 or len(skin_rgb) == 0:
        return {}
    nail_px = nail_rgb.reshape(-1, 1, 3).astype(np.uint8)
    skin_px = skin_rgb.reshape(-1, 1, 3).astype(np.uint8)
    nail_lab = cv2.cvtColor(nail_px, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    skin_lab = cv2.cvtColor(skin_px, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    nail_hsv = cv2.cvtColor(nail_px, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    skin_hsv = cv2.cvtColor(skin_px, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)

    out: dict[str, float] = {}
    for i, ch in enumerate(["l", "a", "b"]):
        n, s = float(np.mean(nail_lab[:, i])), float(np.mean(skin_lab[:, i]))
        out[f"contrast_lab_{ch}_diff"] = n - s
        out[f"contrast_lab_{ch}_ratio"] = n / (s + 1e-6)
    for i, ch in enumerate(["h", "s", "v"]):
        n, s = float(np.mean(nail_hsv[:, i])), float(np.mean(skin_hsv[:, i]))
        out[f"contrast_hsv_{ch}_diff"] = n - s
        out[f"contrast_hsv_{ch}_ratio"] = n / (s + 1e-6)

    def chrom(px_rgb) -> np.ndarray:
        rgb = px_rgb.astype(np.float32)
        return rgb / (rgb.sum(axis=1, keepdims=True) + 1e-6)

    nc, sc = chrom(nail_rgb), chrom(skin_rgb)
    for i, ch in enumerate(["r", "g", "b"]):
        n, s = float(np.mean(nc[:, i])), float(np.mean(sc[:, i]))
        out[f"contrast_chrom_{ch}_diff"] = n - s
        out[f"contrast_chrom_{ch}_ratio"] = n / (s + 1e-6)
    return out


def extended_feature_names() -> list[str]:
    """Feature names produced by extended_features (to expose the column list)."""
    # deterministic order: nail region stats, skin region stats, then contrast
    stats_ch = [f"chrom_{c}_{s}" for c in "rgb" for s in ("mean", "median", "std")]
    stats_ch += [f"hsv_{c}_{s}" for c in ("h", "s", "v") for s in ("mean", "median", "std")]
    stats_ch += ["hsv_h_circular_mean"]
    stats_ch += [f"lab_{c}_{s}" for c in ("l", "a", "b") for s in ("mean", "median", "std")]
    contrast_ch = [f"contrast_{sp}_{c}_{m}" for sp in ("lab", "hsv", "chrom")
                   for c in (("l", "a", "b") if sp == "lab" else (("h", "s", "v") if sp == "hsv" else "rgb"))
                   for m in ("diff", "ratio")]
    return [f"EXT_nail_{n}" for n in stats_ch] + [f"EXT_skin_{n}" for n in stats_ch] + contrast_ch


def extended_features(crop_rgb_norm: np.ndarray, white_median: dict[str, float],
                      nail_mask: np.ndarray | None, skin_crop_norm: np.ndarray | None) -> dict[str, float]:
    """Compute extended features for a nail crop + (optional) full skin crop.

    crop_rgb_norm: white-balanced nail crop (RGB, 0..255) used for nail stats.
    skin_crop_norm: white-balanced skin crop (RGB) used for skin stats and contrast.
    """
    nail_px = crop_rgb_norm[nail_mask > 0] if nail_mask is not None else crop_rgb_norm.reshape(-1, 3)
    skin_px = skin_crop_norm.reshape(-1, 3) if skin_crop_norm is not None else np.array([])

    out = {}
    out.update(region_stats(nail_px, "nail"))
    out.update(region_stats(skin_px, "skin"))
    out.update(contrast_features(nail_px, skin_px))
    return {f"EXT_{k}" if k.startswith(("nail_", "skin_")) else k: v for k, v in out.items()}