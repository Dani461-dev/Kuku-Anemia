"""Generate the printable colour-calibration chart used in every photo.

The chart is an A4 landscape image (3508 x 2480 px @300dpi) containing:
- 4 ArUco markers (DICT_4X4_50, ids 0..3) used to auto-detect the chart
  position/homography in a phone photo;
- a row of flat colour patches with KNOWN sRGB values (white, greys, primaries,
  skin tone). The WHITE patch is used for illumination normalisation.

Outputs (into core/chart_output/):
  chart.png        - printable chart
  chart_refs.json  - known patch colours + canonical markers (needed by
                     calibration.py at detection time)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.config import CHART_DIR, CORE_DIR

CHART_W, CHART_H = 3508, 2480
MARKER_IDS = [0, 1, 2, 3]
MARKER_SIZE = 350
MARGIN = 180
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# name -> sRGB
PATCHES = {
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "black": (25, 25, 25),
    "red": (255, 0, 0),
    "green": (0, 170, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "skin_tone": (222, 184, 150),
    "cyan": (0, 255, 255),
}


def marker_centers() -> dict[int, tuple[int, int]]:
    m = MARGIN + MARKER_SIZE // 2
    return {
        0: (m, m),                                    # top-left
        1: (CHART_W - m, m),                          # top-right
        2: (CHART_W - m, CHART_H - m),                # bottom-right
        3: (m, CHART_H - m),                          # bottom-left
    }


def patch_layout() -> list[tuple[str, tuple[int, int]]]:
    n = len(PATCHES)
    patch_w = 240
    gap = 110
    total = n * patch_w + (n - 1) * gap
    start_x = (CHART_W - total) // 2
    center_y = CHART_H // 2
    out = []
    for i, (name, _rgb) in enumerate(PATCHES.items()):
        cx = start_x + patch_w // 2 + i * (patch_w + gap)
        out.append((name, (cx, center_y)))
    return out


def build_chart() -> np.ndarray:
    img = np.full((CHART_H, CHART_W, 3), 245, np.uint8)  # near-white background
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    for marker_id, (cx, cy) in marker_centers().items():
        top_left = (cx - MARKER_SIZE // 2, cy - MARKER_SIZE // 2)
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE)
        if marker.ndim == 2:
            marker = np.repeat(marker[:, :, None], 3, axis=2)
        img[top_left[1] : top_left[1] + MARKER_SIZE, top_left[0] : top_left[0] + MARKER_SIZE] = marker

    for name, (cx, cy) in patch_layout():
        bgr = PATCHES[name][::-1]  # sRGB stored in refs; cv2 draws in BGR
        half = 110
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), bgr, thickness=-1)
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (0, 0, 0), thickness=3)

    cv2.putText(img, "ANEMIA APP - COLOR REFERENCE", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
    return img


def write_refs_json(path: Path) -> None:
    refs = {
        "chart_px": [CHART_W, CHART_H],
        "aruco_dict": "DICT_4X4_50",
        "marker_size": MARKER_SIZE,
        "markers": {str(mid): list(xy) for mid, xy in marker_centers().items()},
        "patches": {name: {"rgb": list(PATCHES[name]), "center": list(xy)} for name, xy in patch_layout()},
    }
    path.write_text(json.dumps(refs, indent=2))


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart = build_chart()
    png_path = CHART_DIR / "chart.png"
    cv2.imwrite(str(png_path), chart)
    # canonical reference used by calibration.py + a copy alongside the PNG
    refs_path = CORE_DIR / "chart_refs.json"
    write_refs_json(refs_path)
    write_refs_json(CHART_DIR / "chart_refs.json")
    print(f"Chart written to {png_path}")
    print(f"Reference values written to {refs_path}")
    # sanity: chart should be detectable
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY))
    print(f"Sanity detection: {0 if ids is None else len(ids)}/4 ArUco markers found")


if __name__ == "__main__":
    sys.exit(main())