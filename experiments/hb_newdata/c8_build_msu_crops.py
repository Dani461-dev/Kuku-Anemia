#!/usr/bin/env python3
"""C8 — Bangun crop kuku untuk dataset MSU-250 (eval CNN lintas-domain).

Deteksi kuku tengah (seg26) -> crop margin 1.6 -> JPEG ke outputs/crops_msu/.
Sama resep dgn c4_build_crops.py agar CNN dilatih di sewa bisa diuji di MSU.

Usage:
    python experiments/hb_newdata/c8_build_msu_crops.py [--device 0] [--conf 0.15] [--limit 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent            # anemia-app/
sys.path.insert(0, str(ROOT))

from core.pipeline import load_rgb                                # noqa: E402
from core.seg_detector import Yolo26SegDetector                   # noqa: E402
from p0_triage import select_middle_instance                      # noqa: E402

WEIGHTS = ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"
PHOTO_DIR = ROOT / "data" / "photo"
MSU_META = ROOT / "data" / "metadata.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEST = OUT_DIR / "crops_msu"
CSV = OUT_DIR / "crops_msu.csv"


def expand_box(box, img_h: int, img_w: int, margin: float = 1.6):
    t, l, b, r = [float(x) for x in box]
    cy, cx = (t + b) / 2.0, (l + r) / 2.0
    h, w = max(b - t, 20.0), max(r - l, 20.0)
    nh, nw = h * margin, w * margin
    t2 = max(0, int(cy - nh / 2.0)); b2 = min(img_h, int(cy + nh / 2.0))
    l2 = max(0, int(cx - nw / 2.0)); r2 = min(img_w, int(cx + nw / 2.0))
    return t2, l2, b2, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--margin", type=float, default=1.6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(MSU_META)                     # PATIENT_ID, HB_LEVEL_GperL
    DEST.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in DEST.glob("*.jpg")}
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    det = Yolo26SegDetector(weights=WEIGHTS, conf=args.conf, device=args.device)

    n_ok = n_nail = 0
    todo = [r for _, r in meta.iterrows() if str(r["PATIENT_ID"]) not in existing]
    if args.limit:
        todo = todo[:args.limit]
    print(f"MSU crop: total {len(meta)} | sudah ada {len(meta)-len(todo)} | sisa {len(todo)}")
    for r in todo:
        pid = str(r["PATIENT_ID"])
        img = load_rgb(str(PHOTO_DIR / f"{pid}.jpg"))
        h, w = img.shape[:2]
        mid = select_middle_instance(det.segment(img).instances)
        if mid is None:
            n_nail += 1
            continue
        t2, l2, b2, r2 = expand_box(mid.nail_box, h, w, args.margin)
        crop = img[t2:b2, l2:r2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            n_nail += 1
            continue
        out = DEST / f"{pid}.jpg"
        Image.fromarray(crop).save(out, quality=90)
        rows.append({"PATIENT_ID": pid, "MODALITY": "msu",
                     "HB_LEVEL_GperL": float(r["HB_LEVEL_GperL"]),
                     "CROP_PATH": str(out)})
        n_ok += 1
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"selesai: ok={n_ok} no_nail={n_nail} | total log={len(rows)}")


if __name__ == "__main__":
    sys.exit(main())