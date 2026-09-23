#!/usr/bin/env python3
"""C12 — Rekalibrasi & koreksi bias (literatur Mannino 2018/2025, Bihar 2023).

Simulasi jujur di test set:
  - 30% pasien test = "kohort kalibrasi" (punya 1 nilai Hb lab, mis. dari
    protokol data/full_hand) -> dipakai fit koreksi.
  - 70% sisanya = evaluasi (tidak ikut kalibrasi).

Varian koreksi:
  1. global offset   : pred - mean_err(cal)
  2. global linear   : a*pred + b            (Bihar-style retrain ringan)
  3. per-device      : offset/linear terpisah utk A05/A14/S24

Metrik: MAE/R2/Pearson + sens/spec anemia cutoff 120 g/L, sebelum & sesudah.

Usage:
    python experiments/hb_newdata/c12_personalize.py [--tag resnet18_v2] [--cal-frac 0.30] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EXP = Path(__file__).resolve().parent
META = ROOT.parent / "anemia-dataset" / "metadata.csv"
CUTOFF = 120.0         # g/L anemia threshold


def load_patient_preds(tag: str) -> pd.DataFrame:
    """Prediksi per-pasien (rata-rata 2 foto) + per-foto, gabung device/polish."""
    p = pd.read_csv(EXP / "models" / "cnn" / tag / "test_predictions.csv")
    grp = p.groupby("PATIENT_UUID").agg(HB=("HB_LEVEL_GperL", "first"),
                                        PRED=("PRED_GperL", "mean"))
    meta = pd.read_csv(META, dtype=str)[["patient_uuid", "survey_device_modal",
                                         "survey_nail_polish_or_henna"]]
    meta.columns = ["PATIENT_UUID", "DEVICE", "POLISH"]
    df = grp.reset_index().merge(meta, on="PATIENT_UUID", how="left")
    df["DEVICE"] = df["DEVICE"].fillna("UNKNOWN")
    return df


def split_cal_eval(df: pd.DataFrame, cal_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (by device) cal/eval split yang deterministik."""
    cal_idx, ev_idx = [], []
    rng = np.random.RandomState(seed)
    for _, g in df.groupby("DEVICE"):
        ids = g["PATIENT_UUID"].to_numpy()
        rng.shuffle(ids)
        n_cal = int(len(ids) * cal_frac)
        cal_idx += ids[:n_cal].tolist()
        ev_idx += ids[n_cal:].tolist()
    cal = df[df["PATIENT_UUID"].isin(cal_idx)]
    ev = df[df["PATIENT_UUID"].isin(ev_idx)]
    return cal, ev


def report(y, p, tag: str) -> str:
    mae = skm.mean_absolute_error(y, p)
    r2 = skm.r2_score(y, p)
    pear = float(np.corrcoef(y, p)[0, 1])
    auc = float(skm.roc_auc_score((y < CUTOFF).astype(int), -p))  # lower Hb => anemic
    sens = float(skm.recall_score((y < CUTOFF).astype(int), (p < CUTOFF).astype(int)))
    spec = float(skm.recall_score((y >= CUTOFF).astype(int), (p >= CUTOFF).astype(int)))
    return (f"{tag:24s} MAE {mae:6.2f} g/L ({mae/10:.2f} g/dL) | R2 {r2:+6.3f} | "
            f"Pear {pear:5.3f} | AUC {auc:5.3f} | sens {sens:4.2f} spec {spec:4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="resnet18_v2")
    ap.add_argument("--cal-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_patient_preds(args.tag)
    print(f"tag={args.tag} | pasien test={len(df)} | cal_frac={args.cal_frac}")
    print(df["DEVICE"].value_counts().to_string())

    cal, ev = split_cal_eval(df, args.cal_frac, args.seed)
    print(f"\nkalibrasi n={len(cal)} | evaluasi n={len(ev)}")
    y_c, p_c = cal["HB"].to_numpy(), cal["PRED"].to_numpy()
    y_e, p_e = ev["HB"].to_numpy(), ev["PRED"].to_numpy()

    print("\n===== SEBELUM kalibrasi (eval set) =====")
    print(" " + report(y_e, p_e, "asli"))

    rows = []

    # 1) global offset
    off = float((y_c - p_c).mean())
    p1 = p_e + off
    rows.append(("global offset", report(y_e, p1, "global offset")))
    print("\n===== SESUDAH (dipasang di eval set 70%) =====")
    print(" " + rows[-1][1])

    # 2) global linear
    A = np.vstack([p_c, np.ones_like(p_c)]).T
    a, b = np.linalg.lstsq(A, y_c, rcond=None)[0]
    p2 = a * p_e + b
    rows.append(("global linear", report(y_e, p2, "global linear")))
    print(" " + rows[-1][1])

    # 3) per-device offset & linear (bool mask, bukan index)
    p3 = np.empty_like(p_e); p4 = np.empty_like(p_e)
    for dev in ev["DEVICE"].unique():
        m_c = cal[cal["DEVICE"] == dev]
        m_e_mask = (ev["DEVICE"] == dev).to_numpy()
        if len(m_c) < 3 or m_e_mask.sum() < 3:
            p3[m_e_mask] = ev.loc[m_e_mask, "PRED"].to_numpy() + off
            p4[m_e_mask] = a * ev.loc[m_e_mask, "PRED"].to_numpy() + b
            continue
        yc, pc = m_c["HB"].to_numpy(), m_c["PRED"].to_numpy()
        p3[m_e_mask] = ev.loc[m_e_mask, "PRED"].to_numpy() + float((yc - pc).mean())
        Ac = np.vstack([pc, np.ones_like(pc)]).T
        a2, b2 = np.linalg.lstsq(Ac, yc, rcond=None)[0]
        p4[m_e_mask] = a2 * ev.loc[m_e_mask, "PRED"].to_numpy() + b2
    rows.append(("per-device offset", report(y_e, p3, "per-device offset")))
    rows.append(("per-device linear", report(y_e, p4, "per-device linear")))
    print(" " + rows[-1][1])
    print(" " + rows[-2][1])

    # 4) ensemble offset + per-device linear (bias terbesar di literatur: personalisasi)
    #    simplest "1-point personalization" proxy: per-device linear sudah itu.
    print("\n===== ringkasan =====")
    for name, txt in rows:
        print(f"  {txt}")
    print("\ncatatan: kalibrasi memakai label Hb 30% pasien test (simulasi kohort "
          "klinik dengan 1 CBC per pasien). Untuk pemantauan serial 1 pasien, "
          "literatur (Mannino) melaporkan MAE 0.57-0.74 g/dL setelah personalisasi.")


if __name__ == "__main__":
    sys.exit(main())