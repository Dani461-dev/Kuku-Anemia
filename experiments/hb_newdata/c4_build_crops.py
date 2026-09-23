#!/usr/bin/env python3
"""C4a — Bangun dataset crop kuku untuk training CNN.

Untuk tiap pasien × modalitas (open/closed):
  - deteksi kuku dgn seg26 -> ambil instance jari tengah (nail_box [t,l,b,r])
  - perluas box (margin) supaya mencakup kuku + kulit sekitarnya
  - crop -> simpan JPEG ke outputs/crops/{modality}/{uuid}.jpg
  - catat baris (uuid, modality, HB_LEVEL_GperL, path) ke outputs/crops.csv

Resume-friendly: crop yang sudah ada dilewati.

Usage:
    python experiments/hb_newdata/c4_build_crops.py [--modality both] [--device 0] [--margin 1.6] [--limit 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent          # anemia-app/
DATA_ROOT = ROOT.parent / "anemia-dataset"
sys.path.insert(0, str(ROOT))

from core.pipeline import load_rgb                                # noqa: E402
from core.seg_detector import Yolo26SegDetector                   # noqa: E402
from p0_triage import select_middle_instance                      # noqa: E402

WEIGHTS = ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"
META = DATA_ROOT / "metadata.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
CROP_DIR = OUT_DIR / "crops"
CROPS_CSV = OUT_DIR / "crops.csv"


def expand_box(box, img_h: int, img_w: int, margin: float = 1.6):
    """Perluas nail_box [t,l,b,r] dengan faktor `margin` di sekitar pusat."""
    t, l, b, r = [float(x) for x in box]
    cy, cx = (t + b) / 2.0, (l + r) / 2.0
    h, w = max(b - t, 20.0), max(r - l, 20.0)
    nh, nw = h * margin, w * margin
    t2 = max(0, int(cy - nh / 2.0)); b2 = min(img_h, int(cy + nh / 2.0))
    l2 = max(0, int(cx - nw / 2.0)); r2 = min(img_w, int(cx + nw / 2.0))
    return t2, l2, b2, r2


def build(det, meta: pd.DataFrame, modality: str, margin: float, limit: int) -> tuple[int, int, int]:
    mask_col = f"image_{modality}"
    sub = meta[meta[mask_col].notna() & meta["hgb_final"].notna()].reset_index(drop=True)
    if limit:
        sub = sub.head(limit)
    dest = CROP_DIR / modality
    dest.mkdir(parents=True, exist_ok=True)

    # resume
    existing = {p.stem for p in dest.glob("*.jpg")} if dest.exists() else set()
    todo = sub[~sub["patient_uuid"].isin(existing)]

    # load csv log jika ada
    rows = []
    if CROPS_CSV.exists():
        rows = pd.read_csv(CROPS_CSV).to_dict("records")

    n_ok, n_nail, n_skip = 0, 0, 0
    print(f"[{modality}] total {len(sub)} | crop sudah ada {len(sub)-len(todo)} | sisa {len(todo)}")
    for _, r in todo.iterrows():
        uuid = r["patient_uuid"]
        img_path = DATA_ROOT / "images" / modality / r[mask_col]
        hb_gL = float(r["hgb_final"]) * 10.0
        try:
            img = load_rgb(str(img_path))
        except Exception:
            n_skip += 1
            continue
        h, w = img.shape[:2]
        result = det.segment(img)
        mid = select_middle_instance(result.instances)
        if mid is None:
            n_nail += 1
            continue
        t2, l2, b2, r2 = expand_box(mid.nail_box, h, w, margin)
        crop = img[t2:b2, l2:r2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            n_nail += 1
            continue
        out_path = dest / f"{uuid}.jpg"
        Image.fromarray(crop).save(out_path, quality=90)
        rows.append({"PATIENT_UUID": uuid, "MODALITY": modality,
                     "HB_LEVEL_GperL": hb_gL, "CROP_PATH": str(out_path)})
        n_ok += 1
        if n_ok % 500 == 0:
            pd.DataFrame(rows).to_csv(CROPS_CSV, index=False)
            print(f"  [{modality}] {n_ok} crop (log disimpan)")

    pd.DataFrame(rows).to_csv(CROPS_CSV, index=False)
    print(f"[{modality}] selesai: ok={n_ok} no_nail={n_nail} skip={n_skip} | total log={len(rows)}")
    return n_ok, n_nail, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["open", "closed", "both"], default="both")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--margin", type=float, default=1.6, help="faktor perluasan box kuku")
    ap.add_argument("--limit", type=int, default=0, help="0 = semua pasien")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, dtype=str)
    det = Yolo26SegDetector(weights=WEIGHTS, conf=args.conf, device=args.device)
    print(f"margin={args.margin} conf={args.conf} device={args.device}")

    mods = ["fingernails_open", "fingernails_closed"] if args.modality == "both" \
        else ["fingernails_open"] if args.modality == "open" else ["fingernails_closed"]
    for m in mods:
        build(det, meta, m, args.margin, args.limit)


if __name__ == "__main__":
    sys.exit(main())