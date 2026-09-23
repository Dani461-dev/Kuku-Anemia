#!/usr/bin/env python3
"""C7 — Evaluasi ensemble CNN(kuku) + model fitur di test set yang SAMA (seed 42).

Alur:
  1. Rekonstruksi split pasien 80/10/10 seed 42 (sama dgn C4/C6).
  2. Model fitur: RobustScaler+ElasticNet (kanonikal core/train_hb.make_estimator)
     dilatih HANYA di pasien train, memakai fitur mean(open,closed) NAIL_+SKIN_.
     Prediksi pasien val & test.
  3. CNN v1/v2: jalankan inference di crop val & test (TTA opsional) ->
     prediksi per-pasien (rata-rata 2 foto).
  4. Bobot ensemble di-tuning di VAL (grid), dilaporkan di TEST.
  5. Laporan: MAE/R2/Pearson per-pasien, subgrup device & kutek.

Usage:
    python experiments/hb_newdata/c7_eval_ensemble.py [--tta] [--skip-cnn]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.train_hb import make_estimator  # noqa: E402

HB_LO, HB_HI = 30.0, 200.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXP = Path(__file__).resolve().parent
OUT = EXP / "outputs"
META = ROOT.parent / "anemia-dataset" / "metadata.csv"


def split_uuids(seed=42, val_frac=0.10, test_frac=0.10):
    """UUID train/val/test dari crops.csv (harus identik dgn C4/C6)."""
    df = pd.read_csv(OUT / "crops.csv")
    uuids = sorted(df["PATIENT_UUID"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(uuids)
    n_val = int(len(uuids) * val_frac)
    n_test = int(len(uuids) * test_frac)
    val_u, test_u = sorted(uuids[:n_val]), sorted(uuids[n_val:n_val + n_test])
    tr_u = sorted(set(uuids) - set(val_u) - set(test_u))
    return tr_u, val_u, test_u


def load_patient_mean_features():
    """Fitur mean(open,closed) per pasien: kolom NAIL_/SKIN_ + HB."""
    feat_cols = None
    frames = []
    for name in ["fingernails_open", "fingernails_closed"]:
        d = pd.read_csv(OUT / f"features_{name}.csv")
        if feat_cols is None:
            feat_cols = [c for c in d.columns if c.startswith("NAIL_") or c.startswith("SKIN_")]
        keep = ["PATIENT_UUID", "HB_LEVEL_GperL"] + feat_cols
        frames.append(d[keep].set_index("PATIENT_UUID"))
    o, c = frames
    common = o.index.intersection(c.index)
    mean = (o.loc[common] + c.loc[common]) / 2.0
    mean["PATIENT_UUID"] = mean.index
    return mean.reset_index(drop=True), feat_cols


def fit_predict_features(tr_u, va_u, te_u):
    df, feat_cols = load_patient_mean_features()
    df = df[np.isfinite(df[feat_cols].astype(float).values).all(axis=1)].copy()
    hb = df["HB_LEVEL_GperL"].astype(float)
    df = df[hb.between(HB_LO, HB_HI)].copy()
    est = make_estimator(quick=False)
    X = df[feat_cols].astype(float)
    y = df["HB_LEVEL_GperL"].astype(float).to_numpy()
    est.fit(X[df["PATIENT_UUID"].isin(tr_u)], y[df["PATIENT_UUID"].isin(tr_u)])
    out = {}
    for tag, uu in [("val", va_u), ("test", te_u)]:
        m = df[df["PATIENT_UUID"].isin(uu)].set_index("PATIENT_UUID")
        m = m.loc[~m.index.duplicated()]
        p = pd.DataFrame({"PATIENT_UUID": m.index,
                          "HB_LEVEL_GperL": m["HB_LEVEL_GperL"].astype(float).to_numpy(),
                          "PRED_GperL": est.predict(m[feat_cols])})
        out[tag] = p
    best = {"alpha": float(est.named_steps["regressor"].alpha_),
            "l1_ratio": float(est.named_steps["regressor"].l1_ratio_)}
    return out, best


def make_model(arch: str):
    import torchvision.models as tvm
    if arch == "resnet18":
        m = tvm.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(arch)
    return m


class ColorNormalize:
    """Koreksi warna per-gambar — harus SAMA dgn training (meta['colornorm'])."""

    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, img):
        from PIL import Image as _I
        if self.mode == "none":
            return img
        arr = np.asarray(img).astype(np.float32)
        if self.mode == "whitep98":
            refs = np.array([np.percentile(arr[..., c], 98) for c in range(3)])
            if float(refs.min()) < 80.0:
                return img
        else:  # grayworld
            refs = np.array([arr[..., c].mean() for c in range(3)])
            refs = np.clip(refs, 30.0, None)
        out = np.clip(arr / refs[None, None, :] * 255.0, 0, 255)
        return _I.fromarray(out.astype(np.uint8))


def cnn_predict_patients(tag_dir: Path, split_sets, tta: bool, device: str):
    """Inference CNN -> (val_patients, test_patients) prediksi per-pasien (pakai colornorm model)."""
    from torchvision import transforms
    from PIL import Image
    meta = json.loads((tag_dir / "model_metadata.json").read_text())
    arch = "resnet18" if "resnet18" in meta.get("model", "") else \
           ("efficientnet_b0" if "efficientnet" in meta.get("model", "") else "resnet18")
    model = make_model(arch)
    model.load_state_dict(torch.load(tag_dir / "best.pt", map_location="cpu"))
    model.eval()
    model.to(device)
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    cn = ColorNormalize(meta.get("colornorm", "none"))
    ev = transforms.Compose([cn, transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), norm])
    crops = pd.read_csv(OUT / "crops.csv")
    out = {}
    for tag, uu in [("val", split_sets[1]), ("test", split_sets[2])]:
        sub = crops[crops["PATIENT_UUID"].isin(uu)]
        preds, keys = [], []
        with torch.no_grad():
            for _, r in sub.iterrows():
                x = ev(Image.open(r["CROP_PATH"]).convert("RGB")).unsqueeze(0).to(device)
                p = float(model(x).squeeze(0).item())
                if tta:
                    p = 0.5 * (p + float(model(torch.flip(x, [3])).squeeze(0).item()))
                preds.append(p); keys.append(r["PATIENT_UUID"])
        pdf = pd.DataFrame({"PATIENT_UUID": keys, "PRED_GperL": preds})
        out[tag] = pdf.groupby("PATIENT_UUID")["PRED_GperL"].mean().reset_index()
    return out


def report(name, y, p, n=0):
    return f"{name:22s} MAE {skm.mean_absolute_error(y, p):6.2f} g/L ({skm.mean_absolute_error(y, p)/10:.2f} g/dL)" \
           f" | R2 {skm.r2_score(y, p):+6.3f} | Pearson {np.corrcoef(y, p)[0,1]:5.3f}"


def subgroup(df: pd.DataFrame, col: str, label: str):
    rows = []
    for v, g in df.groupby(col):
        rows.append((v, len(g),
                     skm.mean_absolute_error(g["HB"], g["PRED"]),
                     skm.mean_absolute_error(g["HB"], g["PRED"]) / 10))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", action="store_true", help="TTA (flip) saat inference CNN")
    ap.add_argument("--skip-cnn", action="store_true", help="hanya model fitur")
    ap.add_argument("--limit", type=int, default=0, help="batasi pasien val/test (smoke)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_u, va_u, te_u = split_uuids()
    if args.limit:
        va_u, te_u = va_u[:args.limit], te_u[:args.limit]
    print(f"pasien: train {len(tr_u)} | val {len(va_u)} | test {len(te_u)}")

    feat, best = fit_predict_features(tr_u, va_u, te_u)
    print(f"fitur: alpha={best['alpha']:.5f} l1={best['l1_ratio']:.2f}")

    preds = {"feature": feat}
    if not args.skip_cnn:
        for tag_dir in sorted((EXP / "models" / "cnn").iterdir()):
            if not (tag_dir / "best.pt").exists():
                continue
            tag = tag_dir.name
            p = cnn_predict_patients(tag_dir, (tr_u, va_u, te_u), args.tta, device)
            preds[tag] = p
            print(f"CNN {tag}: val/test prediksi selesai (TTA={args.tta})")

    # rangkum val/test per-pasien
    yv = preds["feature"]["val"].set_index("PATIENT_UUID")["HB_LEVEL_GperL"]
    val_frame = pd.DataFrame({"HB": yv})
    models = list(preds.keys())
    for tag in models:
        p = preds[tag]["val"].set_index("PATIENT_UUID")
        val_frame[tag] = p["PRED_GperL"].astype(float)
    tv = pd.DataFrame({"HB": preds["feature"]["test"].set_index("PATIENT_UUID")["HB_LEVEL_GperL"]})
    for tag in models:
        p = preds[tag]["test"].set_index("PATIENT_UUID")
        tv[tag] = p["PRED_GperL"].astype(float)

    print("\n================= VAL (tuning) =================")
    for tag in models:
        print(" " + report(tag, val_frame["HB"], val_frame[tag]))

    # 1) cari bobot ensemble (avg CNN vs fitur) di val: grid 0..1 step 0.05
    cnn_tags = [t for t in models if t != "feature"]
    best_w, best_v = {"cnn": 0.5}, 1e9
    if cnn_tags:
        cnn_avg_v = np.mean([val_frame[t].to_numpy() for t in cnn_tags], axis=0)
        grids = [(w, skm.mean_absolute_error(val_frame["HB"], w * cnn_avg_v + (1 - w) * val_frame["feature"]))
                 for w in np.arange(0.0, 1.001, 0.05)]
        best_w = {"cnn": max(grids, key=lambda x: -x[1])[0]}
        best_v = min(g[1] for g in grids)
        print(f"[val] bobot CNN terbaik = {best_w['cnn']:.2f} (ens CNN+fitur) -> val MAE {best_v:.2f} g/L")

    print("\n================= TEST =================")
    for tag in models:
        print(" " + report(tag, tv["HB"], tv[tag]))
    if len(cnn_tags) >= 1:
        cnn_avg = np.mean([tv[t].to_numpy() for t in cnn_tags], axis=0)
        w = best_w["cnn"]
        ens_p = w * cnn_avg + (1 - w) * tv["feature"]
        print(" " + report(f"ens CNN*{w:.2f}+fitur*{1-w:.2f}", tv["HB"], ens_p))
        if len(cnn_tags) == 2:
            ens12 = (tv[cnn_tags[0]].to_numpy() + tv[cnn_tags[1]].to_numpy()) / 2
            print(" " + report(f"ens {cnn_tags[0]}+{cnn_tags[1]}", tv["HB"], ens12))

    # --- subgrup device & kutek untuk model terbaik ---
    meta = pd.read_csv(META, dtype=str)[["patient_uuid", "survey_device_modal", "survey_nail_polish_or_henna"]]
    meta.columns = ["PATIENT_UUID", "DEVICE", "POLISH"]
    best_p = np.asarray(w * cnn_avg + (1 - w) * tv["feature"].to_numpy()) if len(cnn_tags) >= 1 \
        else tv["feature"].to_numpy()
    res = pd.DataFrame({"PATIENT_UUID": tv.index.to_numpy(), "HB": tv["HB"].to_numpy(), "PRED": best_p})
    res = res.merge(meta, on="PATIENT_UUID", how="left")
    print("\n=== subgrup (model ensemble terbaik) ===")
    for col, lab in [("DEVICE", "device"), ("POLISH", "polish")]:
        for v, n, mae, mae_d in subgroup(res, col, lab):
            print(f"  {lab}={v or '?':<16s} n={n:<4d} MAE {mae:6.2f} g/L ({mae_d:.2f} g/dL)")
    no_polish = res[res["POLISH"] == "No"]
    print(f"  tanpa kutek/henna  n={len(no_polish):<4d} MAE {skm.mean_absolute_error(no_polish['HB'], no_polish['PRED']):.2f} g/L")


if __name__ == "__main__":
    sys.exit(main())