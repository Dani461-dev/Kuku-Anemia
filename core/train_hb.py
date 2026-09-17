#!/usr/bin/env python3
"""Train & save the nail Hb regression model (RobustScaler + ElasticNet).

Two protocols (--protocol):

  nested   [default, the "improved" model]
    - NESTED CV: outer KFold evaluation with an inner ElasticNetCV search
      fitted only on outer training folds -> unbiased error estimate.
    - optionally trains on all 250 patients (--no-balance), which gives the
      best MAE/RMSE in our experiments.

  notebook [reproduces the ORIGINAL GitHub protocol exactly]
    - KDE-balanced 100-patient subset (gaussian_kde bw=0.5, seed 42)
    - GridSearchCV: l1_ratio=[0.01,0.1,0.5,0.9,0.99],
      alpha=logspace(-4,4,100), KFold(7), RMSE scoring
    - expected result: best alpha ~0.205651, l1_ratio 0.9,
      CV RMSE ~23.856 g/L, test RMSE ~20.26 g/L  (matches Usage Notes.ipynb)
    - use the *nomask* feature csv for an exact match:
        python3 core/train_hb.py --protocol notebook \
            --features core/outputs/features_gt_fixed_nomask.csv \
            --out-dir core/models/notebook_baseline

Units: metadata stores Hb in g/L. The model is fit/predicted in g/L.
Metadata ships both g/L and g/dL (= g/L / 10) to match the Anevia response
format, and the WHO anemia thresholds.

Run after core/build_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import scipy.stats as stats  # noqa: E402
import sklearn.linear_model  # noqa: E402
import sklearn.metrics  # noqa: E402
import sklearn.model_selection  # noqa: E402
import sklearn.pipeline  # noqa: E402
import sklearn.preprocessing  # noqa: E402

from core import config as cfg  # noqa: E402
from core.features import feature_names  # noqa: E402

WHO_FEMALE_GPERL = 120.0
WHO_MALE_GPERL = 130.0


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    feat_cols = [f"NAIL_{n}" for n in feature_names()] + [f"SKIN_{n}" for n in feature_names()]
    if any(c.startswith("EXT_") for c in df.columns):
        from core.extended import extended_feature_names

        feat_cols += extended_feature_names()
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"features.csv missing columns: {missing[:5]} ...")
    return df, feat_cols


def balance(df: pd.DataFrame, seed: int = 42, n_subset: int = 100):
    """Return a boolean index selecting the balanced subset (notebook style)."""
    np.random.seed(seed)
    target = df["HB_LEVEL_GperL"].astype(float).values
    kde = stats.gaussian_kde(target, bw_method=0.5)(target)
    weights = 1.0 / kde
    weights /= weights.sum()
    idx = np.random.choice(np.arange(len(df)), size=n_subset, replace=False, p=weights)
    mask = np.zeros(len(df), dtype=bool)
    mask[idx] = True
    return mask


def metrics(y_true, y_pred) -> dict:
    return {
        "mae_gperL": float(sklearn.metrics.mean_absolute_error(y_true, y_pred)),
        "rmse_gperL": float(np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))),
        "r2": float(sklearn.metrics.r2_score(y_true, y_pred)),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def bland_altman(y_true, y_pred) -> dict:
    """Bias prediction - true dan limits of agreement (notebook sel 42)."""
    diff = y_pred - np.asarray(y_true)
    bias = float(np.mean(diff))
    sd = float(np.std(diff))
    return {"bias_gperL": bias, "loa_lo_gperL": bias - 1.96 * sd, "loa_hi_gperL": bias + 1.96 * sd}


def make_estimator(quick: bool) -> sklearn.pipeline.Pipeline:
    if quick:
        alphas = np.logspace(-3, 2, num=20)
        l1_ratios = [0.1, 0.5, 0.9]
    else:
        alphas = np.logspace(-4, 1, num=50)  # same grid as the pipeline doc
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    reg = sklearn.linear_model.ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=5,
        max_iter=10000,
        random_state=42,
        n_jobs=-1,
    )
    return sklearn.pipeline.Pipeline([
        ("scaler", sklearn.preprocessing.RobustScaler()),
        ("regressor", reg),
    ])


def g_to_dl(metrics_gperL: dict, prefix: str = "val") -> dict:
    out = {}
    for k, v in metrics_gperL.items():
        if k == "r2" or k == "pearson":
            out[f"{prefix}_{k}"] = v
        elif k == "mae_gperL":
            out[f"{prefix}_mae_g_dl"] = v / 10.0
        elif k == "rmse_gperL":
            out[f"{prefix}_rmse_g_dl"] = v / 10.0
    return out


def nested_cv(X, y, quick: bool, seed: int = 42):
    """Outer 5-fold; inner ElasticNetCV tuned on each outer train split."""
    outer = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty_like(y, dtype=float)
    t0 = time.time()
    for fold, (tr, va) in enumerate(outer.split(X)):
        est = make_estimator(quick)
        est.fit(X[tr], y[tr])
        pred[va] = est.predict(X[va])
        print(f"  fold {fold + 1}/5 done in {(time.time() - t0):.1f}s")
    return pred, metrics(y, pred)


def notebook_gridsearch(X, y):
    """Exact replica of the original notebook's GridSearchCV protocol."""
    model = sklearn.pipeline.Pipeline([
        ("scaler", sklearn.preprocessing.RobustScaler()),
        ("regressor", sklearn.linear_model.ElasticNet(max_iter=10000)),
    ])
    param_grid = {
        "regressor__l1_ratio": [0.01, 0.1, 0.5, 0.9, 0.99],
        "regressor__alpha": np.logspace(-4, 4, num=100),
    }
    scoring = sklearn.metrics.make_scorer(sklearn.metrics.root_mean_squared_error,
                                          greater_is_better=False)
    cv = sklearn.model_selection.KFold(n_splits=7)
    t0 = time.time()
    gs = sklearn.model_selection.GridSearchCV(
        model, param_grid=param_grid, cv=cv, scoring=scoring,
        return_train_score=True, verbose=1, n_jobs=-1,
    )
    gs.fit(X, y)
    print(f"  GridSearch done in {(time.time() - t0):.1f}s; candidates={len(gs.cv_results_['params'])}")
    return gs


def run_notebook(df, feat_cols, X_all, y_all):
    """Reproduce the original notebook end-to-end (balanced 100 + GridSearchCV)."""
    bal_mask = balance(df)
    X, y = X_all[bal_mask], y_all[bal_mask]
    X_test, y_test = X_all[~bal_mask], y_all[~bal_mask]
    print(f"Training subset: {len(y)} patients (KDE-balanced, notebook protocol)")
    print(f"Hb range: {y.min():.0f}-{y.max():.0f} g/L, <120 g/L frac: {np.mean(y < 120):.2f}")

    print("\n[Notebook GridSearchCV] 7-fold, RMSE scoring:")
    gs = notebook_gridsearch(X, y)
    reg = gs.best_estimator_.named_steps["regressor"]
    print(f"  best alpha={reg.alpha:.12f} l1_ratio={reg.l1_ratio:.2f}")
    print(f"  best CV RMSE (GridSearch) = {-gs.best_score_:.3f} g/L  ({-gs.best_score_ / 10:.2f} g/dL)")

    best_model = gs.best_estimator_
    y_pred = sklearn.model_selection.cross_val_predict(best_model, X, y=y, cv=sklearn.model_selection.KFold(n_splits=7))
    best_model.fit(X, y)
    y_pred_test = best_model.predict(X_test)
    val_metrics = metrics(y, y_pred)
    test_metrics = metrics(y_test, y_pred_test)
    val_ba = bland_altman(y, y_pred)
    test_ba = bland_altman(y_test, y_pred_test)
    print(f"  Val (balanced 100) RMSE = {val_metrics['rmse_gperL']:.2f} g/L")
    print(f"  Test (150)         RMSE = {test_metrics['rmse_gperL']:.2f} g/L")
    print(f"  Val  bias = {val_ba['bias_gperL']:+.1f} g/L, LoA = ({val_ba['loa_lo_gperL']:.1f}, {val_ba['loa_hi_gperL']:.1f}) g/L")
    print(f"  Test bias = {test_ba['bias_gperL']:+.1f} g/L, LoA = ({test_ba['loa_lo_gperL']:.1f}, {test_ba['loa_hi_gperL']:.1f}) g/L")
    return best_model, reg, val_metrics, test_metrics, val_ba, test_ba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=None,
                        help="path to features csv (default: latest core/outputs/features_*.csv)")
    parser.add_argument("--protocol", choices=["nested", "notebook"], default="nested",
                        help="nested = improved (KANONIK, default); notebook = reproduksi notebook asli")
    parser.add_argument("--no-balance", action="store_true", help="(nested only) train on all samples instead of KDE-balanced subset")
    parser.add_argument("--quick", action="store_true", help="smaller hyperparameter grid")
    parser.add_argument("--out-dir", type=Path, default=cfg.MODELS_DIR)
    args = parser.parse_args()

    candidates = sorted(cfg.OUTPUTS_DIR.glob("features_*.csv"))
    nomask = [p for p in candidates if p.stem.endswith("_nomask")]
    if args.features is None:
        # default yang tepat per protokol: nested -> masked, notebook -> nomask
        pool = [p for p in candidates if p not in nomask] if args.protocol == "nested" else nomask
        args.features = pool[-1] if pool else candidates[-1]
    print(f"Using features: {args.features.name}  |  protocol: {args.protocol}")

    df, feat_cols = load_features(args.features)
    X_all = df[feat_cols].astype(float).values
    y_all = df["HB_LEVEL_GperL"].astype(float).values

    assert y_all.min() >= cfg.HB_MIN and y_all.max() <= cfg.HB_MAX, "HB out of g/L range"
    assert np.isfinite(X_all).all(), "features contain NaN/inf"

    if args.protocol == "notebook":
        model, reg, val_metrics, test_metrics, val_ba, test_ba = run_notebook(df, feat_cols, X_all, y_all)
        meta = {
            "model": "RobustScaler + ElasticNet",
            "protocol": "notebook",
            "note": "Exact replica of the original Usage Notes.ipynb (biophotonics-msu/photo-haemoglobin).",
            "features_csv": str(args.features.name),
            "feature_order": feat_cols,
            "n_features": len(feat_cols),
            "best_alpha": float(reg.alpha),
            "best_l1_ratio": float(reg.l1_ratio),
            "cv_rmse_gperL": val_metrics["rmse_gperL"],
            "cv_rmse_g_dl": round(val_metrics["rmse_gperL"] / 10.0, 3),
            "test_rmse_gperL": test_metrics["rmse_gperL"],
            "test_rmse_g_dl": round(test_metrics["rmse_gperL"] / 10.0, 3),
            "val_bias_gperL": val_ba["bias_gperL"],
            "test_bias_gperL": test_ba["bias_gperL"],
            "test_loa_gperL": [test_ba["loa_lo_gperL"], test_ba["loa_hi_gperL"]],
            "n_train": 100,
            "n_test": 150,
        }
        save(model, args, meta, val_metrics=val_metrics, test_metrics=test_metrics)
        return 0

    if args.no_balance:
        X, y = X_all, y_all
        X_test = y_test = None
    else:
        bal_mask = balance(df)
        X, y = X_all[bal_mask], y_all[bal_mask]
        X_test, y_test = X_all[~bal_mask], y_all[~bal_mask]

    print(f"Training subset: {len(y)} patients" + (f", hold-out test: {len(y_test)}" if y_test is not None else ""))
    print(f"Hb range: {y.min():.0f}-{y.max():.0f} g/L, <120 g/L frac: {np.mean(y < 120):.2f}")

    print("\n[Nested CV] outer KFold(5), inner ElasticNetCV:")
    pred_val, val_metrics = nested_cv(X, y, args.quick)
    print(f"  CV MAE  {val_metrics['mae_gperL']:.2f} g/L  ({val_metrics['mae_gperL'] / 10:.2f} g/dL)")
    print(f"  CV RMSE {val_metrics['rmse_gperL']:.2f} g/L  ({val_metrics['rmse_gperL'] / 10:.2f} g/dL)")
    print(f"  CV R²   {val_metrics['r2']:.3f}   Pearson r={val_metrics['pearson']:.3f}")

    # Final estimator on the full training subset
    t0 = time.time()
    model = make_estimator(args.quick)
    model.fit(X, y)
    print(f"\nFinal fit in {(time.time() - t0):.1f}s")
    reg = model.named_steps["regressor"]
    print(f"  best alpha={reg.alpha_:.6f} l1_ratio={reg.l1_ratio_:.3f} "
          f"nonzero_coef={int(np.sum(reg.coef_ != 0))}/{len(reg.coef_)}")

    test_metrics = None
    if X_test is not None:
        y_pred_test = model.predict(X_test)
        test_metrics = metrics(y_test, y_pred_test)
        print(f"\n[Hold-out] MAE {test_metrics['mae_gperL']:.2f} g/L "
              f"({test_metrics['mae_gperL'] / 10:.2f} g/dL), RMSE {test_metrics['rmse_gperL']:.2f}, "
              f"R² {test_metrics['r2']:.3f}, pearson {test_metrics['pearson']:.3f}")

    meta = {
        "model": "RobustScaler + ElasticNet",
        "protocol": "nested",
        "features_csv": str(args.features.name),
        "feature_order": feat_cols,
        "n_features": len(feat_cols),
        "best_alpha": float(reg.alpha_),
        "best_l1_ratio": float(reg.l1_ratio_),
        "n_nonzero_coef": int(np.sum(reg.coef_ != 0)),
        "cv_mae_gperL": val_metrics["mae_gperL"],
        "cv_mae_g_dl": round(val_metrics["mae_gperL"] / 10.0, 3),
        "cv_rmse_gperL": val_metrics["rmse_gperL"],
        "cv_rmse_g_dl": round(val_metrics["rmse_gperL"] / 10.0, 3),
        "cv_r2": val_metrics["r2"],
        "train_balanced": not args.no_balance,
        "n_train": int(len(y)),
    }
    if test_metrics is not None:
        meta["test_mae_gperL"] = test_metrics["mae_gperL"]
        meta["test_rmse_gperL"] = test_metrics["rmse_gperL"]
        meta["test_r2"] = test_metrics["r2"]
        meta["n_test"] = int(len(y_test))
    save(model, args, meta, val_metrics=val_metrics, test_metrics=test_metrics)
    return 0


def save(model, args, meta, val_metrics=None, test_metrics=None) -> None:
    meta.update({
        "who_threshold_female_gperL": WHO_FEMALE_GPERL,
        "who_threshold_male_gperL": WHO_MALE_GPERL,
        "who_threshold_female_g_dl": WHO_FEMALE_GPERL / 10.0,
        "who_threshold_male_g_dl": WHO_MALE_GPERL / 10.0,
        "target_unit": "g/L",
        "api_unit": "g/dL",
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "elasticnet_model.joblib"
    joblib.dump(model, model_path)
    meta_path = args.out_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nModel saved to {model_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    sys.exit(main())