"""Colour-calibration chart detection (ArUco + homography).

Given a photo that contains the printed chart (see chart_generator.py), find
the four ArUco markers, estimate the homography to the canonical chart layout
and sample the median RGB of every known colour patch.

Primary use: the WHITE patch median is the illumination reference used to
normalise features (equivalent of the dataset's fixed white region).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core.config import CHART_REFS_JSON

PATCH_SAMPLE_RADIUS = 100  # pixels (in canonical chart space), must be < 110
MIN_MARKERS = 4


def load_chart_refs(path: Path = CHART_REFS_JSON) -> dict:
    return json.loads(Path(path).read_text())


class ChartDetector:
    def __init__(self, refs: dict | None = None):
        refs = refs or load_chart_refs()
        self.refs = refs
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())

    def detect(self, img_rgb: np.ndarray) -> dict | None:
        """Return {patch_name: {'rgb': [r,g,b], 'center': [x,y]}} or None.

        centres returned in the ORIGINAL image pixel coordinates.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) < MIN_MARKERS:
            return None

        obj_pts, img_pts = [], []
        for i, marker_id in enumerate(ids.flatten().tolist()):
            key = str(int(marker_id))
            if key not in self.refs["markers"]:
                continue
            obj_pts.append(self.refs["markers"][key])
            img_pts.append(tuple(corners[i][0].mean(axis=0)))  # marker centre
        if len(obj_pts) < MIN_MARKERS:
            return None

        H, _ = cv2.findHomography(np.float32(obj_pts), np.float32(img_pts))
        if H is None:
            return None

        patches = {}
        for name, patch in self.refs["patches"].items():
            cx, cy = patch["center"]
            tip = np.float32([[cx, cy]])
            warped = cv2.perspectiveTransform(tip[None, :, :], H)[0][0]
            px, py = int(round(warped[0])), int(round(warped[1]))
            r = PATCH_SAMPLE_RADIUS
            x0, x1 = max(0, px - r), min(img_rgb.shape[1], px + r)
            y0, y1 = max(0, py - r), min(img_rgb.shape[0], py + r)
            region = img_rgb[y0:y1, x0:x1]
            if region.size == 0:
                continue
            med = np.median(region.reshape(-1, 3), axis=0).astype(int)
            patches[name] = {"rgb": [int(med[0]), int(med[1]), int(med[2])], "center": [px, py]}
        return patches or None

    def white_median(self, img_rgb: np.ndarray) -> dict[str, float] | None:
        """Median RGB of the WHITE patch, used for illumination normalisation."""
        patches = self.detect(img_rgb)
        if not patches or "white" not in patches:
            return None
        r, g, b = patches["white"]["rgb"]
        return {"R": r, "G": g, "B": b}