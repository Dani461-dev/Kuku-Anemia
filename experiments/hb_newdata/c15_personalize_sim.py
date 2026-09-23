#!/usr/bin/env python3
"""C15 — Simulasi PERSONALISASI 1-titik di data nyata (Mannino 2018/2025).

Kita punya 2 foto per pasien (fingernails_open & closed) dengan Hb lab SAMA.
Simulasi jujur pemantauan serial:
  - foto 1  = "kunjungan 1": pasien dapat CBC lab -> kalibrasi offset
              (offset = hb_lab - raw_pred_foto1)
  - foto 2  = "kunjungan 2": hanya foto aplikasi -> prediksi terkalibrasi
  Dibuat simetris (foto 1 & 2 saling bergantian jadi kalibrasi/evaluasi).

Model yang disimulasi:
  1. fitur/runtime (ElasticNet, dilatih di pasien train, per-foto) -> paling relevan utk app
  2. CNN v2 (prediksi per-foto dari test_predictions.csv)

Keluaran: MAE per-foto sebelum/sesudah personalisasi + lipat perbaikan.

Usage:
    python experiments/hb_newdata/c15_personalize_sim.py
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


def feature_per_image_raws():
    """Prediksi mentah per-foto (open & closed) dari ElasticNet (runtime-style)."""
    feat_cols = None
    frames = []
    for name in ["fingernails_open", "fingernails_closed"]:
        d = pd.read_csv(OUT / f"features_{name}.csv")
        if feat_cols is None:
            feat_cols = [c for c in d.columns if c.startswith("NAIL_") or c.startswith("SKIN_")]
        keep = ["PATIENT_UUID", "HB_LEVEL_GperL"] + feat_cols
        frames.append(d[keep])
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df[feat_cols].astype(float).values).all(axis=1)].copy()
    df = df[df["HB_LEVEL_GperL"].astype(float).between(HB_LO, HB_HI)].copy()

    tr_u, va_u, te_u = split_uuids()
    est = make_estimator(quick=False)
    X, y = df[feat_cols].astype(float), df["HB_LEVEL_GperL"].astype(float).to_numpy()
    est.fit(X[df["PATIENT_UUID"].isin(tr_u)], y[df["PATIENT_UUID"].isin(tr_u)])
    out = {}
    for uu in te_u:
        m = df[df["PATIENT_UUID"] == uu]
        if len(m) == 0:
            continue
        out[uu] = {"hb": float(m["HB_LEVEL_GperL"].iloc[0]),
                   "raws": list(est.predict(m[feat_cols]).astype(float))}
    return out


def cnn_per_image_raws():
    p = pd.read_csv(CNN_ROOT / "resnet18_v2" / "test_predictions.csv")
    out = {}
    for uu, g in p.groupby("PATIENT_UUID"):
        out[uu] = {"hb": float(g["HB_LEVEL_GperL"].iloc[0]),
                   "raws": list(g["PRED_GperL"].astype(float))}
    return out


def simulate(label: str, data: dict) -> None:
    """raw 1-titik vs personalisasi 1-titik (simetris) — MAE per-foto."""
    raw_errs, cal_errs = [], []
    for uu, d in data.items():
        raws, t = d["raws"], d["hb"]
        if len(raws) < 2:
            continue
        raw_errs += [abs(r - t) for r in raws]
        for i in range(len(raws)):            # tiap foto jadi kalibrasi
            r_cal = raws[i]
            for j in range(len(raws)):
                if j == i:
                    continue
                offset = t - r_cal
                cal_errs.append(abs((raws[j] + offset) - t))
    m_raw = float(np.mean(raw_errs))
    m_cal = float(np.mean(cal_errs))
    print(f"== {label}")
    print(f"   n pasien={len(data)} | foto pasien diuji={len(raw_errs)}")
    print(f"   MAE per-foto      : {m_raw:6.2f} g/L ({m_raw/10:.2f} g/dL)")
    print(f"   MAE personalisasi : {m_cal:6.2f} g/L ({m_cal/10:.2f} g/dL)")
    print(f"   penurunan         : {m_raw-m_cal:+6.2f} g/L  ({100*(1-m_cal/m_raw):+.1f}%)")
    print()


def main():
    print("=" * 72)
    print("SIMULASI PERSONALISASI 1-TITIK (2 foto/pasien, Hb sama)")
    print("=" * 72)
    simulate("fitur/runtime (ElasticNet, per-foto)", feature_per_image_raws())
    simulate("CNN v2 (per-foto)", cnn_per_image_raws())
    print("note: simulasinya 'bersih' (2 foto sesi sama); pemantauan serial "
          "mingguan punya noise ekstra -> gain nyata lebih kecil dari ini.")


if __name__ == "__main__":
    sys.exit(main())