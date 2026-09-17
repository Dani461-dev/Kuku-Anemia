"""Pluggable nail/skin detectors.

Each detector returns a list of NailSkinBoxes (one per finger candidate).
The pipeline then selects the middle finger.

   GTDetector    - ground-truth boxes from metadata.csv (training / baseline)
   YOLODetector  - fine-tuned YOLO11n (trained via experiments/yolo/train.py)
   LightDetector - skin-mask + left-contour heuristics (experimental)

NOTE: as of the restructure, the YOLO detector is NOT trained yet (only a
1-epoch sanity run exists). Constructing it requires experiments/yolo/.../weights/best.pt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CLASS_NAMES, PROJECT_ROOT
from .hand_landmarks import NailSkinBoxes


@dataclass
class DetectedFinger:
    finger_id: int
    nail_box: list[int]
    skin_box: list[int]
    confidence: float
    source: str


class NailSkinDetector:
    name = "base"

    def detect(self, img_rgb: np.ndarray) -> list[DetectedFinger]:
        raise NotImplementedError


class GTDetector(NailSkinDetector):
    name = "gt"

    def __init__(self, metadata_df):
        self._meta = metadata_df

    def detect_for_patient(self, patient_id: int) -> list[DetectedFinger]:
        row = self._meta[self._meta["PATIENT_ID"] == patient_id].iloc[0]
        out = []
        import json

        nails = json.loads(row["NAIL_BOUNDING_BOXES"])
        skins = json.loads(row["SKIN_BOUNDING_BOXES"])
        for i, (nail, skin) in enumerate(zip(nails, skins)):
            out.append(DetectedFinger(finger_id=i, nail_box=[int(v) for v in nail],
                                      skin_box=[int(v) for v in skin], confidence=1.0, source="gt"))
        return out

    def detect(self, img_rgb: np.ndarray) -> list[DetectedFinger]:
        return []  # GTDetector needs patient_id via detect_for_patient


class YOLODetector(NailSkinDetector):
    name = "yolo"

    def __init__(self, weights: str | Path | None = None, conf: float = 0.30, iou: float = 0.5, device: str = "cpu"):
        from ultralytics import YOLO

        weights = weights or PROJECT_ROOT / "experiments" / "yolo" / "runs" / "detector" / "weights" / "best.pt"
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights}. "
                "The YOLO detector is NOT trained yet (see experiments/yolo/train.py). "
                "Use --boxes gt instead, or train the detector first."
            )
        self.conf = conf
        self.iou = iou
        self._model = YOLO(str(weights))

    def detect(self, img_rgb: np.ndarray) -> list[DetectedFinger]:
        results = self._model.predict(img_rgb, conf=self.conf, iou=self.iou, verbose=False, device="cpu")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        h, w = img_rgb.shape[:2]
        nails, skins = [], []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = [int(y1), int(x1), int(y2), int(x2)]  # top, left, bottom, right
            (nails if cls == 0 else skins).append((bbox, conf))

        # pair the closest nail-skin by center distance; nail must be "left of" (smaller x-centre than) skin
        used = set()
        pairs = []
        for n_box, n_conf in nails:
            n_cx = (n_box[1] + n_box[3]) / 2
            best, best_d = None, float("inf")
            for i, (s_box, _) in enumerate(skins):
                if i in used:
                    continue
                s_cx = (s_box[1] + s_box[3]) / 2
                d = abs(n_cx - s_cx) + abs((n_box[0] + n_box[2]) / 2 - (s_box[0] + s_box[2]) / 2)
                if s_cx > n_cx and d < best_d:
                    best_d, best = d, i
            if best is not None:
                used.add(best)
                pairs.append((n_box, skins[best][0], min(n_conf, skins[best][1])))
        pairs.sort(key=lambda p: (p[0][0] + p[0][2]) / 2)  # sort by vertical centre
        return [
            DetectedFinger(finger_id=i, nail_box=n, skin_box=s, confidence=c, source="yolo")
            for i, (n, s, c) in enumerate(pairs)
        ]


class LightDetector(NailSkinDetector):
    """Skin-color + left-contour + distance-transform fingertip heuristic.

    Experimental: assumes fingers point LEFT (the dataset layout). Used only
    as a fallback when no trained detector exists.
    """

    name = "light"

    def __init__(self):
        from .hand_landmarks import HandGeometry

        self.geo = HandGeometry()

    def detect(self, img_rgb: np.ndarray) -> list[DetectedFinger]:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 40, 60), (35, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        hand = max(contours, key=cv2.contourArea)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        md = float(dist.max()) if dist.max() > 0 else 1.0
        _, peaks = cv2.threshold(dist, 0.72 * md, 255, cv2.THRESH_BINARY)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
        tips = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 40:
                continue
            ys, xs = np.where(labels == i)
            d = dist[ys, xs]
            k = int(np.argmax(d))
            tips.append((int(xs[k]), int(ys[k]), float(d[k])))
        tips.sort(key=lambda t: t[0])
        out = []
        for i, (tx, ty, tl) in enumerate(tips[:5]):
            # crude axis-aligned box around the fingertip, inward fingernail
            half = max(int(tl * 0.9), 15)
            nail = [max(0, ty - half), max(0, tx - int(tl * 0.3)), min(img_rgb.shape[0], ty + half),
                    min(img_rgb.shape[1], tx + int(tl * 0.3))]
            skin = [max(0, ty - half), min(img_rgb.shape[1], tx + int(tl * 0.3)),
                    min(img_rgb.shape[0], ty + half),
                    min(img_rgb.shape[1], tx + int(tl * 0.9))]
            out.append(DetectedFinger(finger_id=i, nail_box=nail, skin_box=skin, confidence=0.5, source="light"))
        return out


def make_detector(name: str, **kwargs) -> NailSkinDetector:
    if name == "gt":
        import pandas as pd

        meta = pd.read_csv(PROJECT_ROOT / "data" / "metadata.csv")
        return GTDetector(meta)
    if name == "yolo":
        return YOLODetector(**kwargs)
    if name == "light":
        return LightDetector()
    raise ValueError(f"unknown detector {name!r} (available: gt, yolo, light)")