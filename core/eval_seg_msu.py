#!/usr/bin/env python3
"""Evaluate YOLO26-seg nail-box detection vs MSU ground-truth boxes.

For every MSU photo (data/photo/{PID}.jpg):
  - detect nail instances with Yolo26SegDetector (trained seg26 weights)
  - match predictions to GT NAIL_BOUNDING_BOXES greedily by IoU
  - report detection rate, matched IoU distribution, coverage at IoU>=0.5

Runs at configurable confidence threshold(s) so precision/recall trade-off
can be inspected. MSU layout (3 cut-off fingers entering from the edge)
differs from the V2 training domain (5 nails centred) -> expect lower numbers
here than the on-domain val metrics (mask mAP50 ~0.965).

Usage:
    python3 core/eval_seg_msu.py [--conf 0.30] [--conf 0.15] [--device 0]
                                 [--limit N] [--out CSV]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config as cfg  # noqa: E402
from core.seg_detector import Yolo26SegDetector  # noqa: E402
from core.pipeline import load_rgb  # noqa: E402
from core.util import iou  # noqa: E402

PHOTO_DIR = cfg.PROJECT_ROOT / "data" / "photo"
META = cfg.PROJECT_ROOT / "data" / "legacy_metadata_with_boxes.csv"
# (legacy: berisi NAIL/SKIN_BOUNDING_BOXES utk GT eval)
DEFAULT_WEIGHTS = cfg.PROJECT_ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"


def parse_boxes(raw: str) -> list[list[int]]:
    import json

    return [list(map(int, b)) for b in json.loads(raw)]


def eval_conf(df: pd.DataFrame, weights: Path, conf: float, device: str, limit: int) -> pd.DataFrame:
    det = Yolo26SegDetector(weights=weights, conf=conf, device=device)
    rows = []
    for idx, row in df.iterrows():
        if limit and idx >= limit:
            break
        pid = row["PATIENT_ID"]
        img = load_rgb(str(PHOTO_DIR / f"{pid}.jpg"))
        result = det.segment(img)
        preds = [i.nail_box for i in result.instances]
        gts = parse_boxes(row["NAIL_BOUNDING_BOXES"])

        # greedy match: for each GT pick best unused prediction by IoU
        used = set()
        best_ious = []
        for gt in gts:
            best_i, best_v = -1, 0.0
            for k, p in enumerate(preds):
                if k in used:
                    continue
                v = iou(gt, p)
                if v > best_v:
                    best_v, best_i = v, k
            if best_i >= 0:
                used.add(best_i)
            best_ious.append(best_v)

        rows.append({
            "PATIENT_ID": pid,
            "conf": conf,
            "n_gt": len(gts),
            "n_pred": len(preds),
            "detected": int(len(preds) > 0),
            "matched": int(sum(1 for v in best_ious if v > 0)),
            "matched_iou50": int(sum(1 for v in best_ious if v >= 0.5)),
            "mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
            "box_ious": [round(v, 3) for v in best_ious],
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, conf: float) -> dict:
    sub = df[df["conf"] == conf]
    matched = [v for v in sub["box_ious"] for v in v if v > 0]
    return {
        "conf": conf,
        "n_photos": len(sub),
        "detection_rate": float(sub["detected"].mean()),
        "matched_per_gt": float(sub["matched"].sum() / sub["n_gt"].sum()),
        "cov_iou50": float(sub["matched_iou50"].sum() / sub["n_gt"].sum()),
        "mean_iou_matched": float(np.mean(matched)) if matched else 0.0,
        "median_iou_matched": float(np.median(matched)) if matched else 0.0,
        "mean_n_pred": float(sub["n_pred"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", action="append", type=float, default=[],
                        help="confidence threshold(s); repeatable")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--limit", type=int, default=0, help="limit photos (0=all)")
    parser.add_argument("--out", type=Path, default=cfg.OUTPUTS_DIR / "eval_seg_msu.csv")
    args = parser.parse_args()

    confs = args.conf or [0.30, 0.15]
    df = pd.read_csv(META)
    if not args.weights.exists():
        sys.exit(f"weights not found: {args.weights}")

    frames = [eval_conf(df, args.weights, c, args.device, args.limit) for c in confs]
    all_df = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.out, index=False)

    print(f"weights: {args.weights}")
    for c in confs:
        s = summarize(all_df, c)
        print("\n" + "─" * 58)
        print(f"conf={c:>4}  |  {s['n_photos']} photos")
        print("─" * 58)
        print(f"  detection rate     : {s['detection_rate']:.1%}")
        print(f"  GT matched (any)   : {s['matched_per_gt']:.1%}")
        print(f"  GT matched IoU>=.5 : {s['cov_iou50']:.1%}")
        print(f"  IoU matched mean   : {s['mean_iou_matched']:.3f}")
        print(f"  IoU matched median : {s['median_iou_matched']:.3f}")
        print(f"  avg predictions/img: {s['mean_n_pred']:.1f}")
    print(f"\nfull per-photo csv -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())