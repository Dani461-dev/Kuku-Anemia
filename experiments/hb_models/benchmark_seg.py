#!/usr/bin/env python3
"""Eksperimen regresi Hb — model selain ElasticNet, pada fitur seg-runtime.

Evaluasi seragam: outer KFold(5, shuffle, seed 42), out-of-fold prediksi.
Sumber: core/outputs/features_seg26_seg26_auto.csv (250 pasien x 42 fitur).

Kandidat:
  linear   : ElasticNetCV (baseline) · RidgeCV · Huber
  tree     : RandomForest · ExtraTrees · GradientBoosting · HistGradientBoosting
  gbdt     : XGBoost · LightGBM
  lainnya  : SVR(RBF) · KNN · MLP (kecil)
  ensemble : Stacking (ElasticNet + RandomForest + XGBoost)

Metrik: MAE/RMSE/R2/Pearson (g/L) + hit-rate ±1/±1.5/±2 g/dL.
Aturan deploy: adopsi model alternatif HANYA jika lebih baik >= ADOPT_MIN_MAE (g/L).

Jalankan:
    python3 experiments/hb_models/benchmark_seg.py [--features core/outputs/features_seg26_seg26_auto.csv]
Hasil -> core/outputs/model_benchmark_seg.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor  # noqa: E402
from sklearn.linear_model import ElasticNetCV, HuberRegressor, RidgeCV  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

from core import config as cfg  # noqa: E402
from core.features import feature_names  # noqa: E402

ADOPT_MIN_MAE = 1.0   # g/L improvement required to switch away from ElasticNet


def eval_metrics(y, pred):
    diff = np.abs(pred - y)
    return {
        "mae_gperL": float(mean_absolute_error(y, pred)),
        "rmse_gperL": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "pearson": float(np.corrcoef(y, pred)[0, 1]),
        "hit1": float((diff <= 10).mean()),
        "hit1_5": float((diff <= 15).mean()),
        "hit2": float((diff <= 20).mean()),
    }


def evaluate(name, fit_predict, X, y):
    outer = KFold(5, shuffle=True, random_state=42)
    pred = np.empty_like(y, dtype=float)
    t0 = time.time()
    for tr, va in outer.split(X):
        pred[va] = np.asarray(fit_predict(X[tr], y[tr], X[va])).ravel()
    m = eval_metrics(y, pred)
    print(f"  {name:<20} MAE={m['mae_gperL']:6.2f} RMSE={m['rmse_gperL']:6.2f} "
          f"R2={m['r2']:.3f} r={m['pearson']:.3f} | ±1={m['hit1']*100:4.1f}% "
          f"±1.5={m['hit1_5']*100:4.1f}% ±2={m['hit2']*100:4.1f}%  ({time.time()-t0:.1f}s)")
    return {"name": name, **{k: round(v, 4) for k, v in m.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                        default=cfg.OUTPUTS_DIR / "features_seg26_seg26_auto.csv")
    parser.add_argument("--quick", action="store_true",
                        help="subset kandidat (cepat, untuk cek pipeline)")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    cols = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]
    X_all = df[cols].astype(float).values
    y_all = df["HB_LEVEL_GperL"].astype(float).values
    print(f"Data: {len(y_all)} pasien, {X_all.shape[1]} fitur, Hb {y_all.min():.0f}-{y_all.max():.0f} g/L\n")

    def wrap(make):
        """make() -> estimator; fit on train, predict test."""
        def f(x_tr, y_tr, x_va):
            est = make()
            est.fit(x_tr, y_tr)
            p = est.predict(x_va)
            return p.reshape(-1, 1) if p.ndim == 1 else p
        return f

    res = []

    # --- linear ---
    res.append(evaluate("elasticnet (base)", wrap(
        lambda: Pipeline([("s", RobustScaler()), ("r", ElasticNetCV(max_iter=20000, cv=5,
                                                                    alphas=np.logspace(-4, 1, 40),
                                                                    l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9]))]))
                       , X_all, y_all))
    if not args.quick:
        res.append(evaluate("ridge", wrap(
            lambda: Pipeline([("s", RobustScaler()), ("r", RidgeCV(alphas=np.logspace(-3, 4, 60)))]))
            , X_all, y_all))
        res.append(evaluate("huber", wrap(
            lambda: Pipeline([("s", RobustScaler()), ("r", HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=2000))]))
            , X_all, y_all))
    else:
        print("  (quick: linear subset = elasticnet saja)")

    # --- tree (sklearn) ---
    res.append(evaluate("random_forest", wrap(
        lambda: Pipeline([("s", RobustScaler()), ("r", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1))]))
        , X_all, y_all))
    res.append(evaluate("extra_trees", wrap(
        lambda: Pipeline([("s", RobustScaler()), ("r", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1))]))
        , X_all, y_all))
    if not args.quick:
        res.append(evaluate("grad_boost", wrap(
            lambda: Pipeline([("s", RobustScaler()), ("r", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42))]))
            , X_all, y_all))
    res.append(evaluate("hist_gb", wrap(
        lambda: Pipeline([("s", RobustScaler()), ("r", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=42))]))
        , X_all, y_all))

    # --- GBDT eksternal ---
    try:
        from xgboost import XGBRegressor
        res.append(evaluate("xgboost", wrap(
            lambda: Pipeline([("s", RobustScaler()), ("r", XGBRegressor(n_estimators=300, learning_rate=0.05,
                                                                         max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                                                         random_state=42, n_jobs=-1, verbosity=0))]))
            , X_all, y_all))
    except Exception as e:
        print("  xgboost skip:", e)
    try:
        import lightgbm
        from lightgbm import LGBMRegressor
        res.append(evaluate("lightgbm", wrap(
            lambda: Pipeline([("s", RobustScaler()), ("r", LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                                                          num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                                                          random_state=42, n_jobs=-1, verbose=-1))]))
            , X_all, y_all))
    except Exception as e:
        print("  lightgbm skip:", e)

    # --- lainnya ---
    if not args.quick:
        res.append(evaluate("svr_rbf", wrap(
            lambda: Pipeline([("s", RobustScaler()), ("r", SVR(C=10, epsilon=5))]))
            , X_all, y_all))
        res.append(evaluate("knn", wrap(
            lambda: Pipeline([("s", RobustScaler()), ("r", KNeighborsRegressor(n_neighbors=9, weights="distance"))]))
            , X_all, y_all))
        res.append(evaluate("mlp", wrap(
            lambda: Pipeline([("s", RobustScaler()),
                              ("r", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                                 random_state=42, early_stopping=True))]))
            , X_all, y_all))

    # --- stacking ---
    if not args.quick:
        try:
            base_stack = [
                ("en", Pipeline([("s", RobustScaler()), ("r", ElasticNetCV(max_iter=20000, cv=5))])),
                ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ]
            try:
                from xgboost import XGBRegressor
                base_stack.append(("xgb", XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                       random_state=42, n_jobs=-1, verbosity=0)))
            except Exception:
                pass
            def stack_fit(x_tr, y_tr, x_va):
                st = StackingRegressor(estimators=base_stack,
                                       final_estimator=ElasticNetCV(cv=5, max_iter=20000), cv=5, n_jobs=1)
                st.fit(x_tr, y_tr)
                return st.predict(x_va).reshape(-1, 1)
            res.append(evaluate("stacking", stack_fit, X_all, y_all))
        except Exception as e:
            print("  stacking skip:", e)

    dfr = pd.DataFrame(res).sort_values("mae_gperL")
    out = cfg.OUTPUTS_DIR / "model_benchmark_seg.csv"
    dfr.to_csv(out, index=False)

    print(f"\nRangking by MAE -> {out}")
    print(dfr[["name", "mae_gperL", "rmse_gperL", "r2", "hit1", "hit1_5", "hit2"]].to_string(index=False))

    base_mae = dfr.loc[dfr["name"] == "elasticnet (base)", "mae_gperL"].iloc[0]
    best = dfr.iloc[0]
    best_mae = best["mae_gperL"]
    print("\n" + "─" * 60)
    if best_mae < base_mae - ADOPT_MIN_MAE:
        print(f"ADOPSI: '{best['name']}' lebih baik {base_mae - best_mae:.2f} g/L "
              f"(>= {ADOPT_MIN_MAE} g/L) -> pertimbangkan ganti baseline.")
    else:
        print(f"PERTAHANKAN baseline ElasticNet (selisih best "
              f"{base_mae - best_mae:+.2f} g/L, bawah ambang {ADOPT_MIN_MAE} g/L).")


if __name__ == "__main__":
    sys.exit(main())