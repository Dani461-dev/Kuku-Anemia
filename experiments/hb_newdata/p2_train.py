#!/usr/bin/env python3
"""P2 — Siapkan dataset latih + jalankan retrain ElasticNet (open vs closed vs mean).

Langkah:
  1. Filter fitur valid (finite, Hb ∈ [30,200] g/L = kontrak core/train_hb.py)
  2. Bangun 3 varian dataset di outputs/train/:
       - open  : fingernails_open saja
       - closed: fingernails_closed saja
       - mean  : rata-rata fitur open+closed per pasien
  3. Jalankan core/train_hb.py --protocol nested --no-balance untuk tiap varian
     -> model tersimpan di experiments/hb_newdata/models/{open,closed,mean}/

Usage:
    python experiments/hb_newdata/p2_train.py [--quick]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent   # anemia-app/
OUT_DIR = Path(__file__).resolve().parent / "outputs"
TRAIN_DIR = OUT_DIR / "train"
MODELS_DIR = Path(__file__).resolve().parent / "models"
TRAIN_HB = ROOT / "core" / "train_hb.py"
HB_LO, HB_HI = 30.0, 200.0


def load_features(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / f"features_{name}.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Baris dengan fitur finite + Hb dalam kontrak [30,200] g/L."""
    feat_cols = [c for c in df.columns if c.startswith("NAIL_") or c.startswith("SKIN_")]
    finite = np.isfinite(df[feat_cols].astype(float).values).all(axis=1)
    hb = pd.to_numeric(df["HB_LEVEL_GperL"], errors="coerce")
    in_range = hb.between(HB_LO, HB_HI)
    n0 = len(df)
    df = df[finite & in_range].reset_index(drop=True)
    print(f"  {n0} -> {len(df)} baris (drop NaN fitur / Hb luar {HB_LO:.0f}-{HB_HI:.0f})")
    return df


def build_mean(open_df: pd.DataFrame, closed_df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in open_df.columns if c.startswith("NAIL_") or c.startswith("SKIN_")]
    common = set(open_df["PATIENT_UUID"]) & set(closed_df["PATIENT_UUID"])
    o = open_df[open_df["PATIENT_UUID"].isin(common)].set_index("PATIENT_UUID")
    c = closed_df[closed_df["PATIENT_UUID"].isin(common)].set_index("PATIENT_UUID")
    mean = (o[feat_cols].astype(float) + c[feat_cols].astype(float)) / 2.0
    hb = o[["HB_LEVEL_GperL"]].astype(float)
    merged = mean.join(hb).reset_index()
    merged = merged.rename(columns={"index": "PATIENT_UUID"})
    print(f"  mean(open,closed): {len(merged)} pasien")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="grid lebih kecil (untuk uji cepat)")
    ap.add_argument("--variants", nargs="*", default=["open", "closed", "mean"])
    ap.add_argument("--limit", type=int, default=0,
                    help="batasi baris per varian (0 = semua; untuk smoke test)")
    args = ap.parse_args()

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    open_df = clean(load_features("fingernails_open"))
    closed_df = clean(load_features("fingernails_closed"))
    if args.limit:
        print(f"SMOKE TEST: hanya {args.limit} baris per varian")
        open_df = open_df.head(args.limit)
        closed_df = closed_df.head(args.limit)

    variants = {}
    if "open" in args.variants:
        variants["open"] = open_df
    if "closed" in args.variants:
        variants["closed"] = closed_df
    if "mean" in args.variants:
        variants["mean"] = build_mean(open_df, closed_df)

    summary = []
    for name, df in variants.items():
        csv = TRAIN_DIR / f"features_{name}.csv"
        df.to_csv(csv, index=False)
        print(f"\n=== Train variant: {name} ({len(df)} rows) -> {csv.name} ===")
        out_dir = MODELS_DIR / name
        cmd = [sys.executable, str(TRAIN_HB), "--features", str(csv),
               "--protocol", "nested", "--no-balance", "--out-dir", str(out_dir)]
        if args.quick:
            cmd.append("--quick")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-2500:])
        if r.returncode != 0:
            print("STDERR:", r.stderr[-800:])
        # ambil cv_mae dari metadata
        import json
        meta = json.loads((out_dir / "model_metadata.json").read_text())
        summary.append((name, len(df), meta["cv_mae_gperL"], meta["cv_r2"]))

    print("\n========= RINGKASAN CV =========")
    for name, n, mae, r2 in summary:
        print(f"  {name:8s} n={n:5d}  CV MAE {mae:6.2f} g/L ({mae/10:.2f} g/dL)  R2 {r2:+.3f}")


if __name__ == "__main__":
    sys.exit(main())