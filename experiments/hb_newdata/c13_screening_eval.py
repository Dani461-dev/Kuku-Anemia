#!/usr/bin/env python3
"""C13 — Evaluasi LENSA SCREENING untuk semua kandidat model.

Temuan C12: model CNN punya shrinkage prediksi (range 37-117 g/L padahal
true 18-192) -> MAE oke tapi di cutoff 120 g/L semua orang diprediksi anemia
(sens 1.00, spec 0.00). Script ini mengukur perilaku screening SEMUA kandidat:

  - fitur  (runtime baseline, ElasticNet):  test sewa (reproduksi C7) + MSU
  - CNN v2 (resnet18_v2):                    test sewa + MSU (c9 csv)

Metrik screening per model: range prediksi (min/p10/p90/max), MAE, Pearson,
AUC cutoff 120 g/L, dan sens/spec di cutoff 90/100/110/120 g/L.

Usage:
    python experiments/hb_newdata/c13_screening_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from core.train_hb import make_estimator  # noqa: E402

HB_LO, HB_HI = 30.0, 200.0
EXP = Path(__file__).resolve().parent
OUT = EXP / "outputs"
CNN_ROOT = EXP / "models" / "cnn"


def split_uuids(seed=42, val_frac=0.10, test_frac=0.10):
    df = pd.read_csv(OUT / "crops.csv")
    uuids = sorted(df["PATIENT_UUID"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(uuids)
    n_val, n_test = int(len(uuids) * val_frac), int(len(uuids) * test_frac)
    val_u, test_u = sorted(uuids[:n_val]), sorted(uuids[n_val:n_val + n_test])
    tr_u = sorted(set(uuids) - set(val_u) - set(test_u))
    return tr_u, val_u, test_u


def feature_test_preds():
    """Reproduksi C7: fitur mean(open,closed) -> RobustScaler+ElasticNet di train."""
    feat_cols = None
    frames = []
    for name in ["fingernails_open", "fingernails_closed"]:
        d = pd.read_csv(OUT / f"features_{name}.csv")
        if feat_cols is None:
            feat_cols = [c for c in d.columns if c.startswith("NAIL_") or c.startswith("SKIN_")]
        frames.append(d[["PATIENT_UUID", "HB_LEVEL_GperL"] + feat_cols].set_index("PATIENT_UUID"))
    o, c = frames
    common = o.index.intersection(c.index)
    mean = (o.loc[common] + c.loc[common]) / 2.0
    mean["PATIENT_UUID"] = mean.index
    df = mean.reset_index(drop=True)
    df = df[np.isfinite(df[feat_cols].astype(float).values).all(axis=1)].copy()
    df = df[df["HB_LEVEL_GperL"].astype(float).between(HB_LO, HB_HI)].copy()

    tr_u, va_u, te_u = split_uuids()
    est = make_estimator(quick=False)
    X, y = df[feat_cols].astype(float), df["HB_LEVEL_GperL"].astype(float).to_numpy()
    est.fit(X[df["PATIENT_UUID"].isin(tr_u)], y[df["PATIENT_UUID"].isin(tr_u)])
    te = df[df["PATIENT_UUID"].isin(te_u)].set_index("PATIENT_UUID")
    te = te.loc[~te.index.duplicated()]
    return (te["HB_LEVEL_GperL"].astype(float).to_numpy(),
            est.predict(te[feat_cols]).astype(float))


def screen_report(y, p, label: str, shown: bool = True) -> str:
    mae = skm.mean_absolute_error(y, p)
    pear = float(np.corrcoef(y, p)[0, 1])
    auc = float(skm.roc_auc_score((y < 120).astype(int), -p))
    rg = f"{p.min():.0f}-{p.max():.0f} (p10 {np.percentile(p,10):.0f}/p90 {np.percentile(p,90):.0f})"
    parts = []
    for cc in [90, 100, 110, 120]:
        sens = skm.recall_score((y < cc).astype(int), (p < cc).astype(int))
        spec = skm.recall_score((y >= cc).astype(int), (p >= cc).astype(int))
        parts.append(f"{cc}:S{sens:.2f}/Sp{spec:.2f}")
    if shown:
        print(f"== {label}")
        print(f"   n={len(y)} | true range {y.min():.0f}-{y.max():.0f} "
              f"(p10 {np.percentile(y,10):.0f}/p90 {np.percentile(y,90):.0f})")
        print(f"   pred range {rg}")
        print(f"   MAE {mae:6.2f} g/L ({mae/10:.2f} g/dL) | Pearson {pear:5.3f} "
              f"| AUC@120 {auc:5.3f}")
        print(f"   sens/spec cutoff: " + " | ".join(parts))
        print()
    return f"{label:22s} MAE {mae:6.2f} | Pear {pear:5.3f} | AUC {auc:5.3f} | " + " ".join(parts)


def main():
    rows = []
    print("=" * 78)
    print("LENSA SCREENING — model kandidat aplikasi")
    print("=" * 78)

    # 1) fitur/runtime — sewa test
    y_f, p_f = feature_test_preds()
    rows.append(screen_report(y_f, p_f, "fitur (runtime) — sewa test"))

    # 2) fitur/runtime — MSU (p3 baseline sudah tersimpan)
    msu = pd.read_csv(OUT / "p3_predict_baseline.csv")
    rows.append(screen_report(msu["HB_LEVEL_GperL"].to_numpy(),
                              msu["PRED_GperL"].to_numpy(),
                              "fitur (runtime) — MSU"))

    # 3) CNN v2 — sewa test
    p = pd.read_csv(CNN_ROOT / "resnet18_v2" / "test_predictions.csv")
    g = p.groupby("PATIENT_UUID").agg(HB=("HB_LEVEL_GperL", "first"),
                                      PRED=("PRED_GperL", "mean"))
    rows.append(screen_report(g["HB"].to_numpy(), g["PRED"].to_numpy(),
                              "CNN v2 — sewa test"))

    # 4) CNN v2 — MSU (c9)
    m = pd.read_csv(OUT / "c9_msu_predict_resnet18_v2.csv")
    rows.append(screen_report(m["HB_LEVEL_GperL"].to_numpy(),
                              m["PRED_GperL"].to_numpy(),
                              "CNN v2 — MSU"))

    print("=" * 78)
    print("RINGKASAN (sens/spec per cutoff; S=sens, Sp=spec)")
    print("=" * 78)
    for r in rows:
        print(" " + r)


if __name__ == "__main__":
    sys.exit(main())