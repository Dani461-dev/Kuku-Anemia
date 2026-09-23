#!/usr/bin/env python3
"""P0 — Triage: apakah detektor seg26 + fitur 42-persentil JALAN di foto baru?

Uji pada sampel pasien dari dataset baru (anemia-dataset):
  - fingernails_open  &  fingernails_closed
Ukur: deteksi kuku (%), jumlah instance, mask coverage, fitur finite,
Hb dalam rentang guard (40-180 g/L), dan kesamaan distribusi fitur.

Usage:
    python experiments/hb_newdata/p0_triage.py [--n 300] [--device 0] [--conf 0.15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent          # anemia-app/
DATA_ROOT = ROOT.parent / "anemia-dataset"                   # ../anemia-dataset
sys.path.insert(0, str(ROOT))

from core.features import feature_names                      # noqa: E402
from core.pipeline import load_rgb, patient_features         # noqa: E402
from core.seg_detector import Yolo26SegDetector, nail_skin_masks  # noqa: E402

WEIGHTS = ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"
META = DATA_ROOT / "metadata.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
HB_GL_MIN, HB_GL_MAX = 40.0, 180.0      # guard Hb 4-18 g/dL dalam g/L
FEATURES = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]


def select_middle_instance(instances):
    """y-median instance = jari tengah (mirip build_features_seg)."""
    if not instances:
        return None
    ordered = sorted(instances, key=lambda i: (i.nail_box[0] + i.nail_box[2]) / 2)
    return ordered[len(ordered) // 2]


def extract_one(det, img_rgb: np.ndarray):
    """Extract 42 features + mask coverage for the middle nail instance."""
    result = det.segment(img_rgb)
    mid = select_middle_instance(result.instances)
    if mid is None:
        return None, len(result.instances), None
    nail_mask, skin_mask = nail_skin_masks(mid, img_rgb.shape[0], img_rgb.shape[1])
    boxes = mid.to_detected_finger(img_rgb.shape[0], img_rgb.shape[1])
    feats = patient_features(img_rgb, boxes, white_source="auto",
                             use_mask=True, nail_mask=nail_mask, skin_mask=skin_mask)
    cov = feats.pop("_MASK_COVERAGE", 1.0)
    return feats, len(result.instances), cov


def run_modality(det, meta: pd.DataFrame, modality: str, n: int, seed: int) -> pd.DataFrame:
    mask_col = f"image_{modality}"
    sub = meta[meta[mask_col].notna() & meta["hgb_final"].notna()].reset_index(drop=True)
    sample = sub.sample(n=min(n, len(sub)), random_state=seed)
    rows, skips = [], 0
    for _, r in sample.iterrows():
        uuid = r["patient_uuid"]
        img_path = DATA_ROOT / "images" / modality / r[mask_col]
        hb_gL = float(r["hgb_final"]) * 10.0          # g/dL -> g/L
        try:
            img = load_rgb(str(img_path))
        except Exception as exc:                       # gambar korup/missing
            skips += 1
            continue
        feats, n_inst, cov = extract_one(det, img)
        row = {"PATIENT_UUID": uuid, "MODALITY": modality,
               "HB_LEVEL_GperL": hb_gL, "MASK_COVERAGE": cov,
               "HB_IN_GUARD": HB_GL_MIN <= hb_gL <= HB_GL_MAX,
               "N_INSTANCES": n_inst}
        if feats is None:
            row.update({c: np.nan for c in FEATURES})
            rows.append(row)
            continue
        row.update({c: feats.get(c, 0.0) for c in FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, modality: str) -> None:
    n = len(df)
    det = df["N_INSTANCES"].fillna(0).gt(0)
    nail = df["MASK_COVERAGE"].notna()
    print(f"\n===== {modality} (n={n}) =====")
    print(f"  kuku terdeteksi        : {int(det.sum())} ({det.mean()*100:.1f}%)")
    print(f"  fitur jadi (ada mask)  : {int(nail.sum())} ({nail.mean()*100:.1f}%)")
    if nail.sum():
        cov = df.loc[nail, "MASK_COVERAGE"]
        print(f"  mask coverage          : mean {cov.mean():.3f} | min {cov.min():.3f} "
              f"| <0.05: {int((cov < 0.05).sum())} | >0.99: {int((cov > 0.99).sum())}")
        fdf = df.loc[nail, FEATURES].astype(float)
        finite = np.isfinite(fdf.values).all(axis=1)
        print(f"  semua fitur finite     : {int(finite.sum())} ({finite.mean()*100:.1f}%)")
        # rentang fitur vs training lama (0-1 untuk nilai ternormalisasi)
        vals = fdf.values[np.isfinite(fdf.values)]
        print(f"  rentang nilai fitur    : {vals.min():.4f} .. {vals.max():.4f}")
    hb = pd.to_numeric(df["HB_LEVEL_GperL"], errors="coerce")
    print(f"  Hb (g/L)               : n={hb.notna().sum()} | "
          f"{hb.min():.0f}-{hb.max():.0f} | dalam guard {int(df['HB_IN_GUARD'].sum())} "
          f"({df['HB_IN_GUARD'].mean()*100:.1f}%)")
    out = OUT_DIR / f"p0_{modality}.csv"
    df.to_csv(out, index=False)
    print(f"  -> saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--modality", choices=["open", "closed", "both"], default="both")
    args = ap.parse_args()

    if not WEIGHTS.exists():
        sys.exit(f"seg26 weights not found: {WEIGHTS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, dtype=str)
    print(f"pasien tersedia: {len(meta)} | device={args.device} conf={args.conf} n={args.n}")

    det = Yolo26SegDetector(weights=WEIGHTS, conf=args.conf, device=args.device)
    mods = ["fingernails_open", "fingernails_closed"] if args.modality == "both" \
        else ["fingernails_open"] if args.modality == "open" else ["fingernails_closed"]
    for m in mods:
        df = run_modality(det, meta, m, args.n, args.seed)
        report(df, m)


if __name__ == "__main__":
    sys.exit(main())