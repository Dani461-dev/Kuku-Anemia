#!/usr/bin/env python3
"""Retrain model Hb pada dataset FULL-HAND (T3).

Membaca data/full_hand_metadata.csv (template: data/full_hand_template.csv),
membangun fitur (jalur build_dataset, white=chart), lalu melatih model nested CV
dan menyimpan ke core/models/fullhand_model.joblib (+ metadata).

Jika metadata belum ada → pesan panduan & keluar dengan kode 0.

Jalankan:
    python3 core/train_fullhand.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402

from core import config as cfg  # noqa: E402
from core.build_dataset import build  # noqa: E402
from core.detectors import GTDetector  # noqa: E402
from core.features import feature_names  # noqa: E402
from core.train_hb import balance, make_estimator, metrics, nested_cv  # noqa: E402

FULLHAND_META = cfg.PROJECT_ROOT / "data" / "full_hand_metadata.csv"
FULLHAND_PHOTO = cfg.PROJECT_ROOT / "data" / "full_hand"
WHITE_SOURCE = "chart"


def main() -> None:
    if not FULLHAND_META.exists():
        print(f"[T3] Belum ada {FULLHAND_META}")
        print("     Lihat docs/PROTOKOL_FULLHAND.md untuk protokol pengumpulan.")
        print(f"     Template: data/full_hand_template.csv")
        return 0

    df = pd.read_csv(FULLHAND_META)
    print(f"[T3] Metadata full-hand: {len(df)} subjek")
    missing_cols = [c for c in ("PATIENT_ID", "Hb_LAB_GperL", "NAIL_BOUNDING_BOXES", "SKIN_BOUNDING_BOXES")
                    if c not in df.columns]
    if missing_cols:
        print(f"  ERROR: kolom tidak ada: {missing_cols}")
        return 1

    hb = pd.to_numeric(df["Hb_LAB_GperL"], errors="coerce")
    if hb.isna().any() or not ((hb >= cfg.HB_MIN) & (hb <= cfg.HB_MAX)).all():
        print("  ERROR: Hb_LAB_GperL tidak valid (harus angka, rentang 30–200 g/L).")
        return 1

    if df["NAIL_BOUNDING_BOXES"].astype(str).eq("[]").all():
        print("  ERROR: semua NAIL_BOUNDING_BOXES kosong. Isi manual atau jalankan")
        print("         MediaPipe autodetect + QA dulu (lihat PROTOKOL_FULLHAND.md).")
        return 1

    # target harus bernama HB_LEVEL_GperL agar build_dataset bekerja
    df = df.rename(columns={"Hb_LAB_GperL": "HB_LEVEL_GperL"})

    # ── 1) Fitur (jalur sama dengan runtime: white=chart, mask auto) ──
    print("[T3] Ekstraksi fitur (box dataset, white=chart, mask auto)...")
    detector = GTDetector(df)
    t0 = time.time()
    feats_df = build(df, detector, FULLHAND_PHOTO, WHITE_SOURCE, use_mask=True)
    print(f"  {len(feats_df)} baris fitur dalam {(time.time() - t0):.1f}s")
    if feats_df.empty:
        print("  FAILED: tidak ada fitur terekstrak.")
        return 1

    out_csv = cfg.OUTPUTS_DIR / "features_fullhand_chart.csv"
    feats_df.to_csv(out_csv, index=False)
    print(f"  Fitur -> {out_csv}")

    # ── 2) Training (nested CV, semua data full-hand) ──
    feat_cols = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]
    X = feats_df[feat_cols].astype(float).values
    y = feats_df["HB_LEVEL_GperL"].astype(float).values
    print(f"  Train subset {len(y)} pasien, Hb {y.min():.0f}-{y.max():.0f} g/L")

    pred_val, val_metrics = nested_cv(X, y, quick=False)
    print(f"  CV MAE  {val_metrics['mae_gperL']:.2f} g/L ({val_metrics['mae_gperL'] / 10:.2f} g/dL)")
    print(f"  CV RMSE {val_metrics['rmse_gperL']:.2f} g/L, R² {val_metrics['r2']:.3f}")

    model = make_estimator(quick=False)
    model.fit(X, y)
    reg = model.named_steps["regressor"]

    out_dir = cfg.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "fullhand_model.joblib"
    joblib.dump(model, model_path)

    meta = {
        "model": "RobustScaler + ElasticNet (nested CV)",
        "dataset": "full_hand",
        "features_csv": str(out_csv.name),
        "feature_order": feat_cols,
        "cv_mae_gperL": val_metrics["mae_gperL"],
        "cv_mae_g_dl": round(val_metrics["mae_gperL"] / 10.0, 3),
        "cv_rmse_gperL": val_metrics["rmse_gperL"],
        "cv_r2": val_metrics["r2"],
        "best_alpha": float(reg.alpha_),
        "best_l1_ratio": float(reg.l1_ratio_),
        "n_train": int(len(y)),
        "who_threshold_female_gperL": 120.0,
        "who_threshold_male_gperL": 130.0,
        "target_unit": "g/L",
        "api_unit": "g/dL",
    }
    meta_path = out_dir / "fullhand_model_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Model  -> {model_path}")
    print(f"  Meta   -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())