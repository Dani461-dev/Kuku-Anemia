#!/usr/bin/env python3
"""Build the Hb feature matrix for every patient in metadata.csv.

Modes (--boxes):
  gt     - use ground-truth NAIL_2 / SKIN_2 boxes from metadata.csv (default;
           replicates the original notebook's input space)
  yolo   - use a trained YOLO detector (run experiments/yolo/train.py first); the
           middle detected nail is used -> self-consistent with app inference
  light  - experimental skin-mask heuristic

Output: core/outputs/features.csv (42 features + PATIENT_ID + HB + metadata)
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
from core.detectors import GTDetector, make_detector  # noqa: E402
from core.extended import extended_feature_names  # noqa: E402
from core.features import feature_names  # noqa: E402
from core.pipeline import load_rgb, patient_features, select_middle_finger  # noqa: E402


def build(metadata: pd.DataFrame, detector, photo_dir: Path, white_source: str, use_mask: bool, mask_method: str = "auto", extended: bool = False) -> pd.DataFrame:
    cols = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]
    if extended:
        cols += extended_feature_names()
    rows = []
    failed = []
    ok = 0
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="extracting features"):
        pid = int(row.PATIENT_ID)
        img_path = photo_dir / f"{pid}.jpg"
        if not img_path.exists():
            failed.append((pid, "image missing"))
            continue
        img = load_rgb(str(img_path))
        if isinstance(detector, GTDetector):
            boxes = detector.detect_for_patient(pid)
        else:
            boxes = detector.detect(img)
        middle = select_middle_finger(boxes)
        if middle is None:
            failed.append((pid, "no boxes"))
            continue
        feats = patient_features(img, middle, white_source=white_source, use_mask=use_mask, mask_method=mask_method, extended=extended)
        feats.pop("_PATIENT_ID", None)
        r = {"PATIENT_ID": pid, "DETECTOR": detector.name, "WHITE": white_source,
             "HB_LEVEL_GperL": float(row.HB_LEVEL_GperL), "MASK_COVERAGE": feats.pop("_MASK_COVERAGE", 1.0)}
        for c in cols:
            r[c] = feats.get(c, 0.0)
        rows.append(r)
        ok += 1
    df = pd.DataFrame(rows)
    for pid, reason in failed:
        print(f"  SKIP {pid}: {reason}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes", choices=["gt", "yolo", "light"], default="gt")
    parser.add_argument("--white", choices=["fixed", "chart", "auto"], default="fixed")
    parser.add_argument("--no-mask", action="store_true", help="disable nail masking")
    parser.add_argument("--mask-method", choices=["auto", "kmeans", "otsu", "grabcut"], default="auto")
    parser.add_argument("--extended", action="store_true", help="add HSV/LAB/chromaticity + contrast features")
    parser.add_argument("--out", type=Path, default=cfg.OUTPUTS_DIR / "features.csv")
    args = parser.parse_args()

    metadata = pd.read_csv(cfg.PROJECT_ROOT / "data" / "metadata.csv")
    detector = make_detector(args.boxes)
    df = build(metadata, detector, cfg.PROJECT_ROOT / "data" / "photo",
               args.white, use_mask=not args.no_mask, mask_method=args.mask_method,
               extended=args.extended)

    if df.empty:
        print("FAILED: no features extracted")
        return 1

    # unit guard (g/L dataset range)
    hb = df["HB_LEVEL_GperL"].astype(float)
    assert hb.min() >= cfg.HB_MIN and hb.max() <= cfg.HB_MAX, (
        f"Hb out of expected g/L range [{cfg.HB_MIN},{cfg.HB_MAX}]: "
        f"min={hb.min()}, max={hb.max()} (check units!)"
    )

    tag = f"{args.boxes}_{args.white}" + ("" if not args.no_mask else "_nomask")
    if not args.no_mask and args.mask_method != "auto":
        tag = f"{tag}_{args.mask_method}"
    if args.extended:
        tag = f"{tag}_extended"
    out = args.out.parent / f"{args.out.stem}_{tag}{args.out.suffix}"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(f"Mean mask coverage: {df['MASK_COVERAGE'].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())