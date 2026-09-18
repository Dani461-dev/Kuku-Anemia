#!/usr/bin/env python3
"""Evaluate YOLO26-seg mask quality on the V2 TEST split (GT binary masks).

Per image: union of predicted per-nail masks vs the ground-truth binary mask.
  - Dice (F1 of pixels)
  - IoU
  - precision / recall
  - detection rate

This is the *on-domain* metric (same domain as training) and complements the
MSU cross-domain box IoU (core/eval_seg_msu.py).

Usage:
    python3 core/eval_seg_v2.py [--conf 0.15] [--device 0] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.seg_detector import Yolo26SegDetector  # noqa: E402
from core.pipeline import load_rgb  # noqa: E402

V2 = Path("/mnt/d/Documents/Anemia Kuku/anemia-app/Data Tambahan/archive (12)/NailSegmentationDatasetV2")
TEST_IMG = V2 / "test" / "images"
TEST_MSK = V2 / "test" / "masks"
DEFAULT_WEIGHTS = Path("/mnt/d/Documents/Anemia Kuku/anemia-app/experiments/yolo26_seg/runs/seg26/weights/best.pt")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    s = float(a.sum()) + float(b.sum())
    return 2.0 * inter / s if s > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    imgs = sorted(TEST_IMG.glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]
    det = Yolo26SegDetector(weights=args.weights, conf=args.conf, device=args.device)

    dices, ious, precs, recs, dets, ndets = [], [], [], [], [], []
    for p in imgs:
        img = load_rgb(str(p))
        gt = (cv2.imread(str(TEST_MSK / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE) > 0)
        res = det.segment(img)
        dets.append(int(len(res.instances) > 0))
        ndets.append(len(res.instances))
        if not res or not gt.any():
            dices.append(0.0 if gt.any() else 1.0)
            ious.append(0.0 if gt.any() else 1.0)
            precs.append(0.0)
            recs.append(0.0)
            continue
        pred = np.zeros_like(gt)
        for inst in res.instances:
            pred |= (inst.mask > 0)
        pred = pred & np.ones_like(gt)  # ensure same shape
        tp = float((pred & gt).sum())
        dices.append(dice(pred, gt))
        ious.append(tp / float((pred | gt).sum()) if (pred | gt).sum() else 0.0)
        precs.append(tp / float(pred.sum()) if pred.sum() else 0.0)
        recs.append(tp / float(gt.sum()) if gt.sum() else 0.0)

    print(f"V2 test | conf={args.conf} | {len(imgs)} images")
    print("─" * 52)
    print(f"  detection rate   : {np.mean(dets):.1%}")
    print(f"  mean instances   : {np.mean(ndets):.2f} / image")
    print(f"  mask Dice  mean  : {np.mean(dices):.4f}")
    print(f"  mask IoU   mean  : {np.mean(ious):.4f}")
    print(f"  precision (pixel): {np.mean(precs):.4f}")
    print(f"  recall    (pixel): {np.mean(recs):.4f}")


if __name__ == "__main__":
    sys.exit(main())