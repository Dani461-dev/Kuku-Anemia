#!/usr/bin/env python3
"""Experiment: single-middle-finger vs ensemble of per-finger Hb predictions.

For every MSU photo (has lab ground truth):
  - predict Hb for EACH detected nail instance (YOLO26-seg) using the deployed
    seg_runtime model
  - compare strategies:
      middle        : only the vertical-median instance (current pipeline)
      mean          : mean over ALL instances
      median        : median over ALL instances
      filtered      : median over instances whose features pass the plausibility
                      guard (FEAT_FRAC >= 0.7)
  - report MAE / RMSE / Pearson vs the lab Hb (g/L)

No retraining required; the model is applied to every finger's features.

Usage:
    python3 core/eval_ensemble.py [--conf 0.15] [--device 0] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config as cfg  # noqa: E402
from core.guard import FRAC_FEAT_OK, feature_bounds, feature_plausibility  # noqa: E402
from core.inference import NailHbModel  # noqa: E402
from core.pipeline import load_rgb  # noqa: E402
from core.seg_detector import Yolo26SegDetector, nail_skin_masks  # noqa: E402

PHOTO_DIR = cfg.PROJECT_ROOT / "data" / "photo"
META = cfg.PROJECT_ROOT / "data" / "metadata.csv"
WEIGHTS = cfg.PROJECT_ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"


def metrics(y, p) -> dict:
    diff = y - p
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    pear = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else 0.0
    return {"mae_gperL": round(mae, 2), "rmse_gperL": round(rmse, 2),
            "pearson": round(pear, 4)}


def per_instance_preds(img, instances, model, bounds) -> list[dict]:
    """Predict Hb per detected instance; returns list sorted by nail-box vertical centre."""
    h, w = img.shape[:2]
    items = []
    for i, inst in enumerate(instances):
        mask, skin = nail_skin_masks(inst, h, w)
        boxes = inst.to_detected_finger(h, w)
        try:
            hb = model.predict_masks(img, boxes, nail_mask=mask, skin_mask=skin,
                                     white_source="auto")
            vec = model.features_for_masks(img, boxes, nail_mask=mask,
                                           skin_mask=skin, white_source="auto")
        except Exception:
            continue
        _, frac = feature_plausibility(vec[0], model.feature_order, bounds)
        centre = (inst.nail_box[0] + inst.nail_box[2]) / 2
        items.append((centre, i, hb["estimated_hb_gperL"], frac))
    items.sort(key=lambda t: t[0])           # ascending vertical centre
    return [{"hb_g_l": hb, "feat_frac": frac} for _, _, hb, frac in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()

    meta = pd.read_csv(META)
    det = Yolo26SegDetector(weights=args.weights, conf=args.conf, device=args.device)
    model = NailHbModel()
    bounds = feature_bounds()

    rows = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="ensemble"):
        if args.limit and len(rows) >= args.limit:
            break
        pid = int(row.PATIENT_ID)
        img = load_rgb(str(PHOTO_DIR / f"{pid}.jpg"))
        result = det.segment(img)
        y = float(row.HB_LEVEL_GperL)
        if not result:
            continue
        preds = per_instance_preds(img, result.instances, model, bounds)
        if not preds:
            continue
        mid = preds[len(preds) // 2]                     # vertical median (existing)
        ok = [p for p in preds if p["feat_frac"] >= FRAC_FEAT_OK]

        rows.append({
            "PATIENT_ID": pid, "HB_LAB_GPERL": y, "N": len(preds),
            "middle_gperL": mid["hb_g_l"],
            "mean_gperL": float(np.mean([p["hb_g_l"] for p in preds])),
            "median_gperL": float(np.median([p["hb_g_l"] for p in preds])),
            "filtered_median_gperL": (float(np.median([p["hb_g_l"] for p in ok]))
                                      if ok else np.nan),
        })

    df = pd.DataFrame(rows)
    out = cfg.OUTPUTS_DIR / "eval_ensemble.csv"
    df.to_csv(out, index=False)

    y = df["HB_LAB_GPERL"].values
    print(f"\nMSU photos evaluated: {len(df)} | instances/foto median: "
          f"{int(df['N'].median())}")
    print("─" * 62)
    print(f"{'strategy':<22}{'MAE g/L':>10}{'RMSE g/L':>10}{'Pearson':>10}")
    print("─" * 62)
    for col, name in [("middle_gperL", "middle (current)"),
                      ("mean_gperL", "mean (all fingers)"),
                      ("median_gperL", "median (all fingers)"),
                      ("filtered_median_gperL", "filtered median")]:
        sub = df[df[col].notna()]
        if sub.empty:
            continue
        m = metrics(sub["HB_LAB_GPERL"].values, sub[col].values)
        print(f"{name:<22}{m['mae_gperL']:>10.2f}{m['rmse_gperL']:>10.2f}"
              f"{m['pearson']:>10.4f}")
    print("─" * 62)
    print(f"detail -> {out}")


if __name__ == "__main__":
    sys.exit(main())