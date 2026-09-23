#!/usr/bin/env python3
"""P3 — Validasi lintas-domain: model baru -> prediksi dataset MSU-250.

Menjawab pertanyaan kunci: apakah model yang dilatih pada dataset baru
(sewa-rural-care, foto kuku Samsung) tetap akurat pada foto aplikasi lama
(dataset MSU photo-haemoglobin)?

Input:
  --model     joblib model (RobustScaler+ElasticNet, baru)
  --features  CSV fitur MSU (default core/outputs/features_seg26_seg26_auto.csv)
Output:
  - ringkasan metrik ke stdout
  - predictions_*.csv (PATIENT_ID, HB_LEVEL_GperL, PRED_GperL, ...)

Usage:
    python experiments/hb_newdata/p3_eval_msu.py --model <path.joblib> [--features ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
from sklearn import metrics  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def metrics_text(y, pred, tag: str) -> str:
    mae = metrics.mean_absolute_error(y, pred)
    rmse = float(np.sqrt(metrics.mean_squared_error(y, pred)))
    r2 = metrics.r2_score(y, pred)
    pearson = float(np.corrcoef(y, pred)[0, 1])
    return (f"{tag:28s} MAE {mae:6.2f} g/L ({mae/10:.2f} g/dL) | "
            f"RMSE {rmse:6.2f} | R2 {r2:+.3f} | Pearson {pearson:+.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--features", type=Path,
                    default=ROOT / "core" / "outputs" / "features_seg26_seg26_auto.csv")
    ap.add_argument("--tag", default="model_baru")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.model.with_name("model_metadata.json")) as f:
        meta = json.load(f)
    f_order = meta["feature_order"]
    print(f"model: {meta['model']} | cv_mae {meta.get('cv_mae_gperL')} g/L "
          f"| cv_r2 {meta.get('cv_r2')} | n_train {meta.get('n_train')}")

    model = joblib.load(args.model)
    feats = pd.read_csv(args.features)
    print(f"features MSU: {len(feats)} rows | {args.features.name}")

    X = feats[f_order].astype(float)
    mask = np.isfinite(X.values).all(axis=1) & feats["HB_LEVEL_GperL"].notna()
    print(f"baris valid (fitur finite + Hb ada): {int(mask.sum())}/{len(feats)}")
    X = X[mask]; y = feats.loc[mask, "HB_LEVEL_GperL"].astype(float)

    pred = model.predict(X)
    print(metrics_text(y.to_numpy(), pred, "MSU-250 (" + args.tag + ")"))

    # bandingkan dengan baseline model lama bila ada di metadata
    cv_mae = meta.get("cv_mae_gperL")
    if cv_mae:
        print(f"(bandingkan dgn CV model tsb: MAE {cv_mae} g/L)")
    out = OUT_DIR / f"p3_predict_{args.tag}.csv"
    pd.DataFrame({"PATIENT_ID": feats.loc[mask, "PATIENT_ID"],
                  "HB_LEVEL_GperL": y, "PRED_GperL": pred,
                  "ERR_GperL": pred - y.to_numpy()}).to_csv(out, index=False)
    print(f"-> saved {out}")


if __name__ == "__main__":
    sys.exit(main())