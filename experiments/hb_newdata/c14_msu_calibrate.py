#!/usr/bin/env python3
"""C14 — Nilai rekalibrasi kohort lokal kecil di domain free-bg (MSU).

Bihar-2023: app gagal di domain baru (±4.43 g/dL) -> retrain data lokal
-> ±2.25 g/dL. Pertanyaan deployment: berapa kecil kohort CBC lokal yang
dibutuhkan untuk memperbaiki runtime baseline di domain free-bg?

Data: p3_predict_baseline.csv = prediksi fitur/runtime di 250 MSU.
Simulasi jujur: n_cal pasien acak (seed 42) = kohort (1 CBC/pasien),
fit offset/linear pada pred->Hb, terapkan ke sisa pasien (eval).
Diulang 5 seed untuk interval kepercayaan kasar.

Usage:
    python experiments/hb_newdata/c14_msu_calibrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

EXP = Path(__file__).resolve().parent
OUT = EXP / "outputs"


def report(y, p):
    mae = skm.mean_absolute_error(y, p)
    auc = float(skm.roc_auc_score((y < 120).astype(int), -p))
    sens = skm.recall_score((y < 120).astype(int), (p < 120).astype(int))
    spec = skm.recall_score((y >= 120).astype(int), (p >= 120).astype(int))
    return mae, auc, sens, spec


def main():
    df = pd.read_csv(OUT / "p3_predict_baseline.csv")
    y = df["HB_LEVEL_GperL"].to_numpy().astype(float)
    p = df["PRED_GperL"].to_numpy().astype(float)
    n = len(y)
    print(f"MSU baseline (fitur runtime): n={n}")
    b = report(y, p)
    print(f"  asli        MAE {b[0]:6.2f} g/L | AUC {b[1]:5.3f} | sens {b[2]:4.2f} spec {b[3]:4.2f}")
    print(f"  bias (pred-true): {float((p-y).mean()):+.2f} g/L | slope cov/var: "
          f"{np.cov(y,p)[0,1]/np.var(p):.3f}")
    print()

    print(f"{'n_cal':>5} | {'MAE (g/L)':>12} | {'AUC':>6} | {'sens/spec@120':>14}")
    for n_cal in [10, 20, 40, 60, 80, 125]:
        maes, aucs, sss = [], [], []
        for seed in range(5):
            rng = np.random.RandomState(seed)
            idx = rng.permutation(n)
            cal, ev = idx[:n_cal], idx[n_cal:]
            # linear recalibration fit di kohort, terapkan ke eval
            A = np.vstack([p[cal], np.ones_like(p[cal])]).T
            a, bb = np.linalg.lstsq(A, y[cal], rcond=None)[0]
            p2 = a * p[ev] + bb
            mae, auc, sens, spec = report(y[ev], p2)
            maes.append(mae); aucs.append(auc); sss.append(f"{sens:.2f}/{spec:.2f}")
        print(f"{n_cal:5d} | {np.mean(maes):6.2f}±{np.std(maes):4.2f}  | "
              f"{np.mean(aucs):5.3f} | {sss[0]} (s=0)")

    print()
    print("catatan: n_cal=125 = setengah MSU. Untuk aplikasi, 'kohort lokal' bisa"
          " berupa 20-60 pasien puskesmas dengan CBC saat validasi lapangan.")


if __name__ == "__main__":
    sys.exit(main())