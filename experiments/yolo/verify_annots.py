#!/usr/bin/env python3
"""Draw YOLO labels on sample images to visually verify the annotation
conversion is correct (box positions must match nail & skin regions)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YOLO_DIR = PROJECT_ROOT / "experiments" / "yolo" / "dataset"
OUT_DIR = Path(__file__).resolve().parent / "verification"

CLASS_META = {0: ("NAIL", (0, 0, 255)), 1: ("SKIN", (255, 0, 0))}  # BGR cv2 colors


def draw_one(img_path: Path, label_path: Path, out_path: Path) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    for line in label_path.read_text().strip().splitlines():
        cls, xc, yc, wn, hn = line.split()
        cls = int(cls)
        xc, yc, wn, hn = map(float, (xc, yc, wn, hn))
        x1 = int((xc - wn / 2) * w)
        y1 = int((yc - hn / 2) * h)
        x2 = int((xc + wn / 2) * w)
        y2 = int((yc + hn / 2) * h)
        name, color = CLASS_META[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(str(out_path), img)
    return True


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for split in ("train", "val"):
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split
        for img_path in sorted(img_dir.glob("*.jpg")):
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            out_path = OUT_DIR / f"{split}_{img_path.stem}.jpg"
            if draw_one(img_path, label_path, out_path):
                n += 1
    print(f"Wrote {n} verification images to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    sys.exit(main())