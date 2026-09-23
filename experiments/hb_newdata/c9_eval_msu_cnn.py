#!/usr/bin/env python3
"""C9 — Evaluasi CNN (kuku) lintas-domain di MSU-250.

Untuk tiap model CNN (models/cnn/<tag>/best.pt): inferensi crop MSU
(outputs/crops_msu/crops_msu.csv, hasil c8) -> MAE/R2/Pearson.
Pembanding: baseline model fitur lama MAE 14,93 g/L · R2 0,485.

Usage:
    python experiments/hb_newdata/c9_eval_msu_cnn.py [--tags resnet18 resnet18_v2] [--tta] [--device 0]
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
from PIL import Image
from sklearn import metrics as skm
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXP = Path(__file__).resolve().parent
MSU_CSV = EXP / "outputs" / "crops_msu.csv"
BASELINE = "14.93 g/L | R2 +0.485 | Pearson +0.698"


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
    """Sama dgn training (meta['colornorm']) — wajib agar prediksi sesuai."""

    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, img):
        if self.mode == "none":
            return img
        arr = np.asarray(img).astype(np.float32)
        if self.mode == "whitep98":
            refs = np.array([np.percentile(arr[..., c], 98) for c in range(3)])
            if float(refs.min()) < 80.0:
                return img
        else:
            refs = np.clip(np.array([arr[..., c].mean() for c in range(3)]), 30.0, None)
        out = np.clip(arr / refs[None, None, :] * 255.0, 0, 255)
        return Image.fromarray(out.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--exclude-mix-tag", default=None,
                    help="tag model mix-msu: evaluasi hanya subset MSU holdout (tidak dipakai training)")
    args = ap.parse_args()

    df = pd.read_csv(MSU_CSV)
    print(f"MSU crops: {len(df)} | Hb range {df['HB_LEVEL_GperL'].min():.0f}-{df['HB_LEVEL_GperL'].max():.0f} g/L")
    print(f"baseline (fitur lama): {BASELINE}")

    cnn_root = EXP / "models" / "cnn"
    if args.exclude_mix_tag:
        mx = json.loads((cnn_root / args.exclude_mix_tag / "model_metadata.json").read_text())
        subset_ids = set(str(x) for x in mx.get("msu_test_ids", []))
        if subset_ids:
            df = df[df["PATIENT_ID"].astype(str).isin(subset_ids)]
        print(f"[--exclude-mix-tag {args.exclude_mix_tag}] evaluasi hanya MSU holdout: {len(df)} foto")

    tags = args.tags or [p.name for p in sorted(cnn_root.iterdir()) if (p / "best.pt").exists()]
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    y = df["HB_LEVEL_GperL"].astype(float).to_numpy()
    for tag in tags:
        d = cnn_root / tag
        if not (d / "best.pt").exists():
            print(f"skip {tag}: best.pt tidak ada")
            continue
        meta = json.loads((d / "model_metadata.json").read_text())
        arch = "resnet18" if "resnet18" in meta.get("model", "") else \
               ("efficientnet_b0" if "efficientnet" in meta.get("model", "") else "resnet18")
        cn = ColorNormalize(meta.get("colornorm", "none"))
        ev = transforms.Compose([cn, transforms.Resize(256), transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        model = make_model(arch)
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu"))
        model.eval().to(device)
        preds = []
        with torch.no_grad():
            for _, r in df.iterrows():
                x = ev(Image.open(r["CROP_PATH"]).convert("RGB")).unsqueeze(0).to(device)
                p = float(model(x).squeeze(0).item())
                if args.tta:
                    p = 0.5 * (p + float(model(torch.flip(x, [3])).squeeze(0).item()))
                preds.append(p)
        pred = np.array(preds)
        mae = skm.mean_absolute_error(y, pred)
        rmse = float(np.sqrt(skm.mean_squared_error(y, pred)))
        r2 = skm.r2_score(y, pred)
        pear = float(np.corrcoef(y, pred)[0, 1])
        print(f"\nCNN {tag}  (test MAE sewa {meta.get('test_mae_gperL')} g/L)")
        print(f"  MSU-250        MAE {mae:6.2f} g/L ({mae/10:.2f} g/dL) | RMSE {rmse:6.2f} | "
              f"R2 {r2:+.3f} | Pearson {pear:+.3f}")
        out = EXP / "outputs" / f"c9_msu_predict_{tag}.csv"
        pd.DataFrame({"PATIENT_ID": df["PATIENT_ID"], "HB_LEVEL_GperL": y,
                      "PRED_GperL": pred}).to_csv(out, index=False)
        print(f"  -> saved {out}")


if __name__ == "__main__":
    sys.exit(main())