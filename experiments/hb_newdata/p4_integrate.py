#!/usr/bin/env python3
"""P4 — Pasang model terbaik ke runtime aplikasi (core/models/seg_runtime).

Mengganti model lama yang dipakai inference runtime (NailHbModel memprioritaskan
core/models/seg_runtime). Sebelum menimpa, folder lama di-backup dulu.

Mode aman: default dry-run (tidak mengubah file). Tambahkan --apply untuk eksekusi.

Usage:
    # lihat apa yang akan dilakukan (tanpa mengubah apa pun)
    python experiments/hb_newdata/p4_integrate.py --model-dir experiments/hb_newdata/models/mean

    # eksekusi sungguhan
    python experiments/hb_newdata/p4_integrate.py --model-dir experiments/hb_newdata/models/mean --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # anemia-app/
SEG_RUNTIME = ROOT / "core" / "models" / "seg_runtime"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="folder berisi elasticnet_model.joblib + model_metadata.json (mis. .../models/mean)")
    ap.add_argument("--apply", action="store_true", help="jalankan penggantian (tanpa ini hanya dry-run)")
    args = ap.parse_args()

    model_file = args.model_dir / "elasticnet_model.joblib"
    meta_file = args.model_dir / "model_metadata.json"
    for f in (model_file, meta_file):
        if not f.exists():
            sys.exit(f"file tidak ada: {f}")

    meta = json.loads(meta_file.read_text())
    print(f"Model baru : {args.model_dir}")
    print(f"  protokol  : {meta.get('protocol')} | n_train {meta.get('n_train')} | "
          f"features {meta.get('features_csv')}")
    print(f"  CV MAE    : {meta.get('cv_mae_gperL')} g/L ({meta.get('cv_mae_g_dl')} g/dL) | "
          f"CV R2 {meta.get('cv_r2')}")

    if not SEG_RUNTIME.exists():
        sys.exit(f"folder runtime tidak ada: {SEG_RUNTIME}")

    # informasi model yang sedang terpasang
    cur_meta = SEG_RUNTIME / "model_metadata.json"
    if cur_meta.exists():
        cm = json.loads(cur_meta.read_text())
        print(f"Model lama : {SEG_RUNTIME}")
        print(f"  protokol  : {cm.get('protocol')} | n_train {cm.get('n_train')} | "
              f"CV MAE {cm.get('cv_mae_gperL')} g/L | CV R2 {cm.get('cv_r2')}")

    if not args.apply:
        print("\n[dry-run] tidak ada file yang diubah. Jalankan dengan --apply untuk mengganti.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = SEG_RUNTIME.with_name(f"seg_runtime_backup_{stamp}")
    shutil.copytree(SEG_RUNTIME, backup)
    print(f"\nBackup lama -> {backup}")

    shutil.copy2(model_file, SEG_RUNTIME / "elasticnet_model.joblib")
    shutil.copy2(meta_file, SEG_RUNTIME / "model_metadata.json")
    print(f"Dipasang: {SEG_RUNTIME}/elasticnet_model.joblib + model_metadata.json")
    print("Selesai. Model runtime sekarang memakai model baru.")
    return 0


if __name__ == "__main__":
    sys.exit(main())