#!/usr/bin/env python3
"""P1 — Ekstraksi fitur skala penuh pada dataset baru (anemia-dataset).

Untuk setiap pasien (punya foto + hgb_final):
  - seg26 (YOLO26-seg) -> mask kuku + skin geometri
  - white=auto + mask -> 42 fitur persentil (NAIL_/SKIN_)
  - target HB_LEVEL_GperL (hgb_final g/dL * 10)

Resume-friendly: pasien yang sudah ada di output dilewati.
Output: experiments/hb_newdata/outputs/features_{modality}.csv

Usage:
    python experiments/hb_newdata/p1_build_features.py \
        [--modality both] [--limit 0] [--device 0] [--conf 0.15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT.parent / "anemia-dataset"
sys.path.insert(0, str(ROOT))

from core.features import feature_names                      # noqa: E402
from core.pipeline import load_rgb, patient_features         # noqa: E402
from core.seg_detector import Yolo26SegDetector, nail_skin_masks  # noqa: E402
from p0_triage import select_middle_instance, extract_one    # noqa: E402

WEIGHTS = ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"
META = DATA_ROOT / "metadata.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
FEATURES = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]
CHUNK = 200


def build(det, meta: pd.DataFrame, modality: str, limit: int, out_csv: Path) -> int:
    mask_col = f"image_{modality}"
    sub = meta[meta[mask_col].notna() & meta["hgb_final"].notna()].reset_index(drop=True)
    if limit:
        sub = sub.head(limit).reset_index(drop=True)

    # resume: pasien yang sudah ada
    done: set[str] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv, dtype=str)
        if not prev.empty and "PATIENT_UUID" in prev.columns:
            done = set(prev["PATIENT_UUID"])
        print(f"  resume: {len(done)} pasien sudah ada di {out_csv.name}")

    todo = sub[~sub["patient_uuid"].isin(done)]
    print(f"[{modality}] total {len(sub)} | sisa diproses {len(todo)}")
    if todo.empty:
        return len(done)

    rows, saved = [], 0
    for _, r in todo.iterrows():
        uuid = r["patient_uuid"]
        img_path = DATA_ROOT / "images" / modality / r[mask_col]
        hb_gL = float(r["hgb_final"]) * 10.0
        try:
            img = load_rgb(str(img_path))
        except Exception:
            saved += 1
            continue
        feats, n_inst, cov = extract_one(det, img)
        row = {"PATIENT_UUID": uuid, "MODALITY": modality,
               "HB_LEVEL_GperL": hb_gL, "MASK_COVERAGE": cov, "N_INSTANCES": n_inst}
        if feats is None:
            row.update({c: np.nan for c in FEATURES})
        else:
            row.update({c: feats.get(c, 0.0) for c in FEATURES})
        rows.append(row)
        if len(rows) >= CHUNK:
            flush(rows, out_csv)
            saved += len(rows)
            rows = []
            print(f"  [{modality}] {saved}/{len(todo)} diproses")
    if rows:
        flush(rows, out_csv)
        saved += len(rows)
    print(f"[{modality}] selesai: {saved} diproses | total di CSV = {saved + len(done)}")
    return saved + len(done)


def flush(rows: list, out_csv: Path):
    df = pd.DataFrame(rows)
    header = not out_csv.exists()
    df.to_csv(out_csv, mode="a", header=header, index=False)


def summary(out_csv: Path, modality: str) -> None:
    if not out_csv.exists():
        return
    df = pd.read_csv(out_csv)
    nail = df["MASK_COVERAGE"].notna()
    print(f"\n===== {modality} ({out_csv.name}) =====")
    print(f"  total baris    : {len(df)}")
    print(f"  ada fitur      : {int(nail.sum())} ({nail.mean()*100:.1f}%)")
    if nail.sum():
        fdf = df.loc[nail, FEATURES].astype(float)
        finite = np.isfinite(fdf.values).all(axis=1)
        print(f"  fitur finite   : {int(finite.sum())} ({finite.mean()*100:.1f}%)")
        cov = df.loc[nail, "MASK_COVERAGE"]
        print(f"  mask coverage  : mean {cov.mean():.3f} | <0.05: {int((cov < 0.05).sum())} "
              f"| >0.99: {int((cov > 0.99).sum())}")
    hb = pd.to_numeric(df["HB_LEVEL_GperL"], errors="coerce")
    print(f"  Hb g/L range   : {hb.min():.0f} - {hb.max():.0f} | n={hb.notna().sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["open", "closed", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="0 = semua pasien")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, dtype=str)
    det = Yolo26SegDetector(weights=WEIGHTS, conf=args.conf, device=args.device)

    mods = ["fingernails_open", "fingernails_closed"] if args.modality == "both" \
        else ["fingernails_open"] if args.modality == "open" else ["fingernails_closed"]
    for m in mods:
        out = OUT_DIR / f"features_{m}.csv"
        build(det, meta, m, args.limit, out)
        summary(out, m)


if __name__ == "__main__":
    sys.exit(main())