#!/usr/bin/env python3
"""C10 — Bangun crop SEMUA jari (bukan hanya jari tengah) untuk multi-finger.

Untuk tiap pasien x modalitas: semua instance kuku dari seg26 -> crop margin 1.6
-> JPEG outputs/crops_all/{modality}/{uuid}_f{i}.jpg + baris crops_all.csv.

Guna: melatih/model inference dengan rata-rata prediksi semua kuku pasien
(lebih banyak sinyal daripada 1 kuku tengah).

Usage:
    python experiments/hb_newdata/c10_build_all_crops.py [--device 0] [--conf 0.15] [--margin 1.6] [--limit 0]
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

from core.pipeline import load_rgb                              # noqa: E402
from core.seg_detector import Yolo26SegDetector                 # noqa: E402

WEIGHTS = ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"
META = DATA_ROOT / "metadata.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEST = OUT_DIR / "crops_all"
CSV = OUT_DIR / "crops_all.csv"
MIN_SIZE = 24          # crop sekecil ini dianggap bukan kuku, dilewati
MAX_PER_PHOTO = 6      # amankan: maks instance per foto (biasanya 5)


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

    meta = pd.read_csv(META, dtype=str)
    DEST.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(CSV).to_dict("records") if CSV.exists() else []
    done_uuids = {r["PATIENT_UUID"] for r in rows}
    det = Yolo26SegDetector(weights=WEIGHTS, conf=args.conf, device=args.device)

    mods = ["fingernails_open", "fingernails_closed"]
    todo = [r for _, r in meta.iterrows()
            if r["patient_uuid"] not in done_uuids
            and all(pd.notna(r[f"image_{m}"]) for m in mods)
            and pd.notna(r["hgb_final"])]
    if args.limit:
        todo = todo[:args.limit]
    print(f"semua-jari crop: total pasien {len(todo)} | sudah dilog {len(done_uuids)}")

    n_ok = n_noclip = 0
    for i, r in enumerate(todo, 1):
        uuid = r["patient_uuid"]
        hb_gL = float(r["hgb_final"]) * 10.0
        for m in mods:
            img_path = DATA_ROOT / "images" / m / r[f"image_{m}"]
            img = load_rgb(str(img_path))
            h, w = img.shape[:2]
            insts = det.segment(img).instances
            insts = sorted(insts, key=lambda x: x.confidence, reverse=True)[:MAX_PER_PHOTO]
            for fi, inst in enumerate(insts, 1):
                t2, l2, b2, r2 = expand_box(inst.nail_box, h, w, args.margin)
                crop = img[t2:b2, l2:r2]
                if crop.size == 0 or crop.shape[0] < MIN_SIZE or crop.shape[1] < MIN_SIZE:
                    n_noclip += 1
                    continue
                out = DEST / m / f"{uuid}_f{fi}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop).save(out, quality=90)
                rows.append({"PATIENT_UUID": uuid, "MODALITY": m, "FINGER_ID": fi,
                             "HB_LEVEL_GperL": hb_gL, "CROP_PATH": str(out),
                             "CONF": float(inst.confidence)})
                n_ok += 1
        if i % 800 == 0:
            pd.DataFrame(rows).to_csv(CSV, index=False)
            print(f"  {i} pasien -> {n_ok} crop (log disimpan)")

    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"selesai: {len(todo)} pasien, {n_ok} crop, skip kecil={n_noclip}, total log={len(rows)}")


if __name__ == "__main__":
    sys.exit(main())