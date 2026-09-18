#!/usr/bin/env python3
"""Build the Hb feature matrix using the RUNTIME seg26 feature path.

Same extraction as production runtime:
  YOLO26-seg mask (nail) + geometric skin box + white=auto + 42 percentiles.

metadata.csv is used ONLY for patient-id / Hb lab (g/L). No GT boxes are read.
This makes training features identical in distribution to runtime features
("re-baseline" alignment) without needing new lab data.

Run AFTER training seg26 (experiments/yolo26_seg/runs/seg26/weights/best.pt).

Output: core/outputs/features_seg26_auto.csv
        (PATIENT_ID, DETECTOR, WHITE, HB_LEVEL_GperL, MASK_COVERAGE, 42 features)

Usage:
    python3 core/build_features_seg.py [--conf 0.15] [--device 0]
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
from core.features import feature_names  # noqa: E402
from core.pipeline import load_rgb, patient_features  # noqa: E402
from core.seg_detector import Yolo26SegDetector, nail_skin_masks  # noqa: E402

PHOTO_DIR = cfg.PROJECT_ROOT / "data" / "photo"
META = cfg.PROJECT_ROOT / "data" / "metadata.csv"
WEIGHTS = cfg.PROJECT_ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"


def select_middle_instance(instances):
    """Vertical-median instance (y-median of nail box) = the app's middle finger."""
    if not instances:
        return None
    ordered = sorted(instances, key=lambda i: (i.nail_box[0] + i.nail_box[2]) / 2)
    return ordered[len(ordered) // 2]


def validate_photo_metadata(meta, photo_dir: Path) -> None:
    """1:1 contract: every metadata row has exactly one photo; report orphans."""
    missing, orphans = [], []
    photo_ids = set()
    for p in photo_dir.glob("*.jpg"):
        try:
            photo_ids.add(int(p.stem))
        except ValueError:
            continue
    meta_ids = set(meta["PATIENT_ID"].astype(int))
    missing = sorted(meta_ids - photo_ids)
    orphans = sorted(photo_ids - meta_ids)
    if missing:
        raise SystemExit(f"PHOTO MISSING untuk pasien: {missing[:10]} (n={len(missing)})")
    if orphans:
        print(f"WARN: foto tanpa metadata (tidak dipakai): {orphans[:10]} (n={len(orphans)})")


def build_features(meta: pd.DataFrame, photo_dir: Path, det,
                   white: str = "auto") -> pd.DataFrame:
    """Extract seg26 features for every metadata row (1 foto per pasien).

    Detector 'det' = Yolo26SegDetector instance. White source should match the
    deployed Hb model (auto untuk seg_runtime). Returns DataFrame with
    PATIENT_ID, DETECTOR, WHITE, HB_LEVEL_GperL, MASK_COVERAGE + 42 features.
    """
    cols = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]
    validate_photo_metadata(meta, photo_dir)
    rows, failed = [], []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="seg26 features"):
        pid = int(row.PATIENT_ID)
        img_path = photo_dir / f"{pid}.jpg"
        if not img_path.exists():
            failed.append((pid, "image missing"))
            continue
        img = load_rgb(str(img_path))
        result = det.segment(img)
        mid = select_middle_instance(result.instances)
        if mid is None:
            failed.append((pid, "no nail detected"))
            continue

        nail_mask, skin_mask = nail_skin_masks(mid, img.shape[0], img.shape[1])
        boxes = mid.to_detected_finger(img.shape[0], img.shape[1])
        feats = patient_features(img, boxes, white_source=white,
                                 use_mask=True, nail_mask=nail_mask,
                                 skin_mask=skin_mask)
        feats.pop("_PATIENT_ID", None)
        r = {"PATIENT_ID": pid, "DETECTOR": "seg26", "WHITE": white,
             "HB_LEVEL_GperL": float(row.HB_LEVEL_GperL),
             "MASK_COVERAGE": feats.pop("_MASK_COVERAGE", 1.0)}
        for c in cols:
            r[c] = feats.get(c, 0.0)
        rows.append(r)

    df = pd.DataFrame(rows)
    for pid, reason in failed:
        print(f"  SKIP {pid}: {reason}")
    if df.empty:
        sys.exit("FAILED: no features extracted")
    hb = df["HB_LEVEL_GperL"].astype(float)
    assert hb.min() >= cfg.HB_MIN and hb.max() <= cfg.HB_MAX, "Hb out of g/L range"
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.15,
                        help="seg26 confidence threshold")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--white", choices=["auto", "fixed", "chart"], default="auto",
                        help="white source used in the runtime feature path")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--meta", type=Path, default=META,
                        help="(optional) metadata path (default: data/metadata.csv)")
    parser.add_argument("--photo-dir", type=Path, default=PHOTO_DIR,
                        help="(optional) photo dir (default: data/photo)")
    parser.add_argument("--out", type=Path, default=cfg.OUTPUTS_DIR / "features_seg26.csv")
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f"seg26 weights not found: {args.weights}")

    meta = pd.read_csv(args.meta)
    det = Yolo26SegDetector(weights=args.weights, conf=args.conf, device=args.device)
    df = build_features(meta, args.photo_dir, det, white=args.white)

    cols = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]
    finite = np.isfinite(df[cols].astype(float).values).all()
    print(f"\nrows: {len(df)} | subjects: {int(df['PATIENT_ID'].nunique())}")
    print(f"all features finite: {finite}")
    print(f"mask coverage mean: {df['MASK_COVERAGE'].mean():.3f} "
          f"(min {df['MASK_COVERAGE'].min():.3f}, max {df['MASK_COVERAGE'].max():.3f})")
    hb = df["HB_LEVEL_GperL"].astype(float)
    print(f"Hb range: {hb.min():.0f}-{hb.max():.0f} g/L")

    tag = f"seg26_{args.white}"
    out = args.out.parent / f"{args.out.stem}_{tag}{args.out.suffix}"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")


if __name__ == "__main__":
    sys.exit(main())