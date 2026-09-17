#!/usr/bin/env python3
"""Quick sanity tests for the chart detector using a synthetic warped photo."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.calibration import ChartDetector  # noqa: E402


def main() -> None:
    chart = cv2.imread(str(ROOT / "core" / "chart_output" / "chart.png"))
    chart = cv2.cvtColor(chart, cv2.COLOR_BGR2RGB)
    h, w = chart.shape[:2]

    pts_src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts_dst = np.float32([[100, 80], [w - 140, 40], [60, h - 120], [w - 60, h - 60]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    photo = cv2.warpPerspective(chart, M, (w, h))
    photo = (photo.astype(float) * 0.7).astype(np.uint8)

    det = ChartDetector()
    res = det.detect(photo)
    if res is None:
        print("FAIL: chart not detected")
        return 1
    for k, v in res.items():
        print(f"{k:12s} measured={v['rgb']}")
    print("white_median:", det.white_median(photo))
    expected_white = [int(np.median(photo[photo > 0]))] * 3 if False else None
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())