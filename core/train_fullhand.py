#!/usr/bin/env python3
"""Retrain model Hb pada dataset FULL-HAND (T3) — jalur YOLO-seg.

Membaca data/full_hand_metadata.csv (template: data/full_hand_template.csv,
lean: PATIENT_ID + Hb_LAB_GperL; tanpa box — region dari YOLO26-seg):
  1) Ekstraksi fitur runtime (seg mask + skin geometri + white=auto) per foto.
  2) Training nested CV dengan 2 strategi:
       A) full-hand saja
       B) MSU (features_seg26_seg26_auto.csv) + full-hand  →  pilih terbaik
  3) Simpan model terbaik (fullhand_model.joblib) + metadata.

Jika metadata belum ada / foto kosong → pesan panduan & keluar bersih.

Jalankan:
    python3 core/train_fullhand.py [--conf 0.15] [--device 0]
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
from core.build_features_seg import build_features, validate_photo_metadata  # noqa: E402
from core.features import feature_names  # noqa: E402
from core.seg_detector import Yolo26SegDetector  # noqa: E402
from core.train_hb import make_estimator, metrics, nested_cv  # noqa: E402

FULLHAND_META = cfg.PROJECT_ROOT / "data" / "full_hand_metadata.csv"
FULLHAND_PHOTO = cfg.PROJECT_ROOT / "data" / "full_hand"
MSU_FEATURES = cfg.OUTPUTS_DIR / "features_seg26_seg26_auto.csv"
SEG26_WEIGHTS = cfg.PROJECT_ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"


def prepare_data(conf: float, device: str) -> pd.DataFrame:
    """Extract seg features for every full-hand subject (or exit point)."""
    meta = pd.read_csv(FULLHAND_META)
    print(f"[T3] Metadata full-hand: {len(meta)} subjek")
    missing_cols = [c for c in ("PATIENT_ID", "Hb_LAB_GperL") if c not in meta.columns]
    if missing_cols:
        sys.exit(f"  ERROR: kolom wajib tidak ada: {missing_cols}")

    hb = pd.to_numeric(meta["Hb_LAB_GperL"], errors="coerce")
    if hb.isna().any() or not ((hb >= cfg.HB_MIN) & (hb <= cfg.HB_MAX)).all():
        sys.exit("  ERROR: Hb_LAB_GperL tidak valid (harus angka, rentang 30-200 g/L).")

    meta = meta.rename(columns={"Hb_LAB_GperL": "HB_LEVEL_GperL"})
    validate_photo_metadata(meta, FULLHAND_PHOTO)
    det = Yolo26SegDetector(weights=SEG26_WEIGHTS, conf=conf, device=device)
    t0 = time.time()
    feats = build_features(meta, FULLHAND_PHOTO, det, white="auto")
    print(f"  {len(feats)} baris fitur dalam {(time.time() - t0):.1f}s")
    if feats.empty:
        sys.exit("  FAILED: tidak ada fitur terekstrak.")
    out = cfg.OUTPUTS_DIR / "features_fullhand_seg26_auto.csv"
    feats.to_csv(out, index=False)
    print(f"  Fitur -> {out}")
    return feats


def train_and_report(name: str, X, y) -> dict:
    print(f"\n[{name}] train {len(y)} pasien, Hb {y.min():.0f}-{y.max():.0f} g/L")
    pred_val, val_metrics = nested_cv(X, y, quick=False)
    print(f"  CV MAE  {val_metrics['mae_gperL']:.2f} g/L ({val_metrics['mae_gperL'] / 10:.2f} g/dL)")
    print(f"  CV RMSE {val_metrics['rmse_gperL']:.2f} g/L, R² {val_metrics['r2']:.3f}, "
          f"Pearson {val_metrics['pearson']:.3f}")

    model = make_estimator(quick=False)
    model.fit(X, y)
    reg = model.named_steps["regressor"]
    return {
        "name": name, "model": model,
        "n_train": int(len(y)),
        "cv_mae_gperL": val_metrics["mae_gperL"],
        "cv_mae_g_dl": round(val_metrics["mae_gperL"] / 10.0, 3),
        "cv_rmse_gperL": val_metrics["rmse_gperL"],
        "cv_r2": val_metrics["r2"],
        "cv_pearson": val_metrics["pearson"],
        "best_alpha": float(reg.alpha_), "best_l1_ratio": float(reg.l1_ratio_),
    }


def save_model(meta: dict, out_name: str) -> None:
    model = meta.pop("model")
    out_dir = cfg.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{out_name}.joblib"
    metadata = {
        "model": "RobustScaler + ElasticNet (nested CV)",
        "dataset": "full_hand (+MSU)",
        "feature_pipeline": "seg26_mask+geo_skin",
        "white_source": "auto",
        "feature_order": meta["feature_order"],
        **{k: v for k, v in meta.items() if k != "feature_order"},
        "who_threshold_female_gperL": 120.0,
        "who_threshold_male_gperL": 130.0,
        "target_unit": "g/L", "api_unit": "g/dL",
    }
    joblib.dump(model, model_path)
    (out_dir / f"{out_name}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  Model -> {model_path}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    if not FULLHAND_META.exists():
        print(f"[T3] Belum ada {FULLHAND_META}")
        print("     Lihat docs/PROTOKOL_FULLHAND.md untuk protokol pengumpulan.")
        print(f"     Template: data/full_hand_template.csv")
        return 0
    imgs = list(FULLHAND_PHOTO.glob("*.jpg")) if FULLHAND_PHOTO.exists() else []
    if not imgs:
        print("[T3] data/full_hand/ belum berisi foto. Taruh foto {PID}.jpg sesuai metadata.")
        return 0

    feats = prepare_data(args.conf, args.device)
    feat_cols = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]

    # A) full-hand saja
    X_fh = feats[feat_cols].astype(float).values
    y_fh = feats["HB_LEVEL_GperL"].astype(float).values
    res_a = train_and_report("full-hand only", X_fh, y_fh)

    # B) MSU + full-hand
    if MSU_FEATURES.exists():
        msu = pd.read_csv(MSU_FEATURES)
        X_m = msu[feat_cols].astype(float).values
        y_m = msu["HB_LEVEL_GperL"].astype(float).values
        X_c = np.vstack([X_m, X_fh]); y_c = np.concatenate([y_m, y_fh])
        res_b = train_and_report("MSU+full-hand (gabung)", X_c, y_c)
    else:
        res_b = None

    # pilih terbaik by CV MAE (lebih kecil lebih baik)
    candidates = [res_a] + ([res_b] if res_b else [])
    best = min(candidates, key=lambda r: r["cv_mae_gperL"])
    print(f"\nPILIHAN: {best['name']} (CV MAE {best['cv_mae_gperL']:.2f} g/L)")

    strategy = best["name"]
    if strategy == "full-hand only":
        final = res_a
        if res_b:  # simpan juga gabungan sebagai alternatif
            save_model(res_b, "fullhand_msu_combined")
    else:
        final = res_b
    save_model(final, "fullhand_model")
    print("\n[T3] Selesai. Model full-hand disimpan; gunakan di inference "
          "(env ANEVIA_HB_MODEL_DIR=core/models utk memilih, atau salin ke core/models/).")
    return 0


if __name__ == "__main__":
    sys.exit(main())