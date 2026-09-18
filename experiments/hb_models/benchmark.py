#!/usr/bin/env python3
"""Eksperimen akurasi regresi Hb — perbandingan konfigurasi (nested CV, OOF).

Protokol evaluasi seragam = outer KFold(5, shuffle, seed 42), out-of-fold
prediksi pada 250 pasien. Metrik: MAE/RMSE/R2/Pearson (g/L & g/dL) + hit-rate
±1, ±1.5, ±2 g/dL.

Kandidat:
  1 base        ElasticNetCV (kanonik sekarang)
  2 tuned       ElasticNetCV grid lebih luas
  3 ridge       l1_ratio kecil (near-ridge)
  4 log        ElastikNet pada sqrt(Hb) (target transform, evaluasi kembali g/L)
  5 bag_kde     rata-rata N model @ subset KDE-balanced (variance reduction)
  6 aug         augmentasi fitur (gaussian noise) saat fit
  7 ensemble    rata-rata prediksi ElasticNet + Ridge + Huber

Jalankan:
    python3 experiments/hb_models/benchmark.py [--features core/outputs/features_gt_fixed.csv]
Hasil CSV -> core/outputs/experiment_results.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import scipy.stats as stats  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

from core import config as cfg  # noqa: E402
from core.features import feature_names  # noqa: E402
from core.train_hb import metrics  # noqa: E402


def eval_metrics(y, pred):
    m = metrics(y, pred)
    diff = np.abs(pred - y)
    m["hit1"] = float((diff <= 10).mean())   # <=1 g/dL
    m["hit1_5"] = float((diff <= 15).mean())  # <=1.5 g/dL
    m["hit2"] = float((diff <= 20).mean())   # <=2 g/dL
    return m


def kde_balanced_subset(y_all, size, seed):
    np.random.seed(seed)
    kde = stats.gaussian_kde(y_all, bw_method=0.5)(y_all)
    w = 1.0 / kde
    w /= w.sum()
    idx = np.random.choice(np.arange(len(y_all)), size=size, replace=False, p=w)
    m = np.zeros(len(y_all), bool)
    m[idx] = True
    return m


def evaluate(name, fit_predict, X, y):
    outer = KFold(5, shuffle=True, random_state=42)
    pred = np.empty_like(y, dtype=float)
    t0 = time.time()
    for tr, va in outer.split(X):
        pred[va] = np.asarray(fit_predict(X[tr], y[tr], X[va])).ravel()
    met = eval_metrics(y, pred)
    print(f"  {name:<12} MAE={met['mae_gperL']:5.2f} RMSE={met['rmse_gperL']:5.2f} "
          f"R2={met['r2']:.3f} r={met['pearson']:.3f} | ±1={met['hit1']*100:4.1f}% "
          f"±1.5={met['hit1_5']*100:4.1f}% ±2={met['hit2']*100:4.1f}%  ({time.time()-t0:.1f}s)")
    return {"name": name, **{k: round(v, 4) for k, v in met.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=cfg.OUTPUTS_DIR / "features_gt_fixed.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    cols = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]
    X_all = df[cols].astype(float).values
    y_all = df["HB_LEVEL_GperL"].astype(float).values
    print(f"Data: {len(y_all)} pasien, {X_all.shape[1]} fitur, Hb {y_all.min():.0f}-{y_all.max():.0f} g/L\n")

    ys = y_all.reshape(-1, 1)
    res = []

    # 1 base (ElasticNetCV standar)
    def base():
        from core.train_hb import make_estimator
        est = make_estimator(False)
        def f(x_tr, y_tr, x_va):
            est.fit(x_tr, y_tr); return est.predict(x_va)
        return f
    res.append(evaluate("base", base(), X_all, y_all))

    # 2 tuned (grid luas)
    def tuned():
        reg = Pipeline([("s", RobustScaler()), ("r", ElasticNet(max_iter=20000))])
        from sklearn.model_selection import GridSearchCV
        gs = GridSearchCV(reg, {"r__l1_ratio": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99],
                                "r__alpha": np.logspace(-4, 1.5, 60)}, cv=KFold(5, shuffle=True, random_state=42),
                          scoring="neg_mean_absolute_error", n_jobs=-1)
        def f(x_tr, y_tr, x_va):
            gs.fit(x_tr, y_tr); return gs.predict(x_va)
        return f
    res.append(evaluate("tuned", tuned(), X_all, y_all))

    # 3 ridge
    def ridge():
        reg = Pipeline([("s", RobustScaler()), ("r", Ridge())])
        from sklearn.model_selection import GridSearchCV
        gs = GridSearchCV(reg, {"r__alpha": np.logspace(-3, 4, 60)}, cv=KFold(5, shuffle=True, random_state=42),
                          scoring="neg_mean_absolute_error", n_jobs=-1)
        def f(x_tr, y_tr, x_va):
            gs.fit(x_tr, y_tr); return gs.predict(x_va)
        return f
    res.append(evaluate("ridge", ridge(), X_all, y_all))

    # 4 sqrt target
    def log():
        est = Pipeline([("s", RobustScaler()), ("r", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=20000))])
        def f(x_tr, y_tr, x_va):
            est.fit(x_tr, np.sqrt(y_tr)); return est.predict(x_va) ** 2
        return f
    res.append(evaluate("sqrt_tgt", log(), X_all, y_all))

    # 5 bagging KDE-balanced subset
    def bag(n_models=6, size=120):
        def f(x_tr, y_tr, x_va):
            preds = np.zeros((len(x_va),))
            for s in range(n_models):
                m = kde_balanced_subset(y_tr, size, seed=1000 + s)
                est = Pipeline([("s", RobustScaler()),
                                ("r", ElasticNetCV_alpha())])
                est.fit(x_tr[m], y_tr[m])
                preds += est.predict(x_va)[:, 0] if False else est.predict(x_va)
            return (preds / n_models).reshape(-1, 1)
        return f
    try:
        from sklearn.linear_model import ElasticNetCV
        def make_el():
            return ElasticNetCV(l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
                                alphas=np.logspace(-4, 1.5, 40), cv=3, max_iter=10000, random_state=42)
        from sklearn.pipeline import Pipeline as _P
        def bag():
            def runner(x_tr, y_tr, x_va):
                preds = np.zeros(len(x_va))
                for s in range(6):
                    m = kde_balanced_subset(y_tr, 120, seed=1000 + s)
                    est = _P([("s", RobustScaler()), ("r", make_el())])
                    est.fit(x_tr[m], y_tr[m])
                    preds += est.predict(x_va)
                return (preds / 6).reshape(-1, 1)
            return runner
        res.append(evaluate("bag_kde", bag(), X_all, y_all))
    except Exception as e:
        print("  bag_kde gagal:", e)

    # 6 augmentation (gaussian noise, 2x duplication)
    def augment():
        est = Pipeline([("s", RobustScaler()), ("r", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000))])
        def f(x_tr, y_tr, x_va):
            noise = np.random.RandomState(7).normal(0, np.std(x_tr, axis=0, keepdims=True) * 0.05,
                                                    (len(x_tr), x_tr.shape[1]))
            Xa = np.vstack([x_tr, x_tr + noise])
            ya = np.concatenate([y_tr, y_tr])
            est.fit(Xa, ya)
            return est.predict(x_va)
        return f
    res.append(evaluate("aug_noise", augment(), X_all, y_all))

    # 7 ensemble (avg ElasticNet + Ridge + Huber)
    def ensemble():
        def f(x_tr, y_tr, x_va):
            pipes = [
                Pipeline([("s", RobustScaler()), ("r", ElasticNet(alpha=0.02, l1_ratio=0.3, max_iter=20000))]),
                Pipeline([("s", RobustScaler()), ("r", Ridge(alpha=20.0))]),
                Pipeline([("s", RobustScaler()), ("r", HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=2000))]),
            ]
            preds = np.zeros(len(x_va))
            for p in pipes:
                p.fit(x_tr, y_tr)
                preds += p.predict(x_va)
            return (preds / len(pipes)).reshape(-1, 1)
        return f
    res.append(evaluate("ensemble3", ensemble(), X_all, y_all))

    dfr = pd.DataFrame(res).sort_values("mae_gperL")
    out = cfg.OUTPUTS_DIR / "experiment_results.csv"
    dfr.to_csv(out, index=False)
    print(f"\nRangking by MAE -> {out}")
    print(dfr[["name", "mae_gperL", "rmse_gperL", "r2", "hit1", "hit1_5", "hit2"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())