#!/usr/bin/env python3
"""C4b — Training CNN regresi Hb dari crop kuku (ResNet18 / EfficientNet-B0).

Menggunakan foto crop kuku (outputs/crops.csv) langsung dari piksel —
menembus plateau fitur 42-persentil manual. Pretrained ImageNet, fine-tune,
loss SmoothL1 (Huber) pada Hb (g/L). Split per PASIENTA (80/10/10) agar foto
open+closed pasien yang sama tidak bocor antar set.

Output: models/cnn/{arch}/best.pt + model_metadata.json + test_predictions.csv

Usage:
    python experiments/hb_newdata/c4_train_cnn.py [--arch resnet18] [--epochs 20] [--batch 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core import config as cfg  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropsDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["CROP_PATH"].tolist()
        self.y = df["HB_LEVEL_GperL"].astype(float).to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, float(self.y[i])


def make_model(arch: str, pretrained: bool = True):
    import torchvision.models as tvm
    if arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return m


def split_by_patient(df: pd.DataFrame, seed: int, val_frac: float, test_frac: float):
    """Split per pasien (uuid unik) -> row-level index."""
    uuids = sorted(df["PATIENT_UUID"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(uuids)
    n_val = int(len(uuids) * val_frac)
    n_test = int(len(uuids) * test_frac)
    val_u, test_u = uuids[:n_val], uuids[n_val:n_val + n_test]
    tr_u = set(uuids) - set(val_u) - set(test_u)
    idx = df["PATIENT_UUID"]
    return (df[idx.isin(tr_u)].index, df[idx.isin(val_u)].index,
            df[idx.isin(test_u)].index, val_u, test_u)


def make_transforms(train: bool):
    from torchvision import transforms
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.10, hue=0.02),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), norm,
    ])


def metrics(y, pred):
    from sklearn import metrics as skm
    return {"mae_gperL": float(skm.mean_absolute_error(y, pred)),
            "rmse_gperL": float(np.sqrt(skm.mean_squared_error(y, pred))),
            "r2": float(skm.r2_score(y, pred)),
            "pearson": float(np.corrcoef(y, pred)[0, 1])}


def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device)).squeeze(1)
            ys.append(y.numpy()); preds.append(out.float().cpu().numpy())
    ys = np.concatenate(ys); preds = np.concatenate(preds)
    return metrics(ys, preds), ys, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, default=Path(__file__).resolve().parent / "outputs" / "crops.csv")
    ap.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="0 = semua crop (untuk smoke test)")
    ap.add_argument("--freeze-backbone", action="store_true", help="backbone beku, hanya head (uji cepat)")
    args = ap.parse_args()

    if not args.crops.exists():
        sys.exit(f"crops.csv tidak ada: {args.crops} — jalankan c4_build_crops.py dulu")
    df = pd.read_csv(args.crops)
    df = df[df["CROP_PATH"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"crop: {len(df)} gambar | {df['MODALITY'].value_counts().to_dict()} | arch={args.arch}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"device: {device}")

    tr_idx, va_idx, te_idx, _, _ = split_by_patient(df, args.seed, 0.10, 0.10)
    tr_df = df.loc[tr_idx].reset_index(drop=True)
    va_df = df.loc[va_idx].reset_index(drop=True)
    te_df = df.loc[te_idx].reset_index(drop=True)
    print(f"split per pasien | train {len(tr_df)} | val {len(va_df)} | test {len(te_df)}")

    ds_tr = CropsDataset(tr_df, make_transforms(True))
    ds_va = CropsDataset(va_df, make_transforms(False))
    ds_te = CropsDataset(te_df, make_transforms(False))
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=0)

    model = make_model(args.arch)
    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in (model.fc.parameters() if hasattr(model, "fc") else
                  model.classifier.parameters()):
            p.requires_grad = True
    model.to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           patience=3, factor=0.5)
    scaler = torch.amp.GradScaler("cuda")

    out_dir = Path(__file__).resolve().parent / "models" / "cnn" / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)
    best_mae, best_state, best_epoch, no_improve = 1e9, None, -1, 0

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_b = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                out = model(x).squeeze(1)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach()) * len(y); n_b += len(y)
        val_m, _, _ = evaluate(model, dl_va, device)
        scheduler.step(val_m["mae_gperL"])
        el = time.time() - t0
        print(f"epoch {ep:2d}/{args.epochs} | train_loss {loss_sum/max(n_b,1):.3f} | "
              f"val MAE {val_m['mae_gperL']:.2f} g/L (R2 {val_m['r2']:+.3f}) | {el/60:.1f} mnt")
        if val_m["mae_gperL"] < best_mae:
            best_mae, best_epoch, no_improve = val_m["mae_gperL"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop di epoch {ep} (patience {args.patience})")
                break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cpu"))
    model.to(device)
    te_m, te_y, te_p = evaluate(model, dl_te, device)
    print("\n===== TEST (set terpisah, per pasien) =====")
    print(f"MAE {te_m['mae_gperL']:.2f} g/L ({te_m['mae_gperL']/10:.2f} g/dL) | "
          f"RMSE {te_m['rmse_gperL']:.2f} | R2 {te_m['r2']:+.3f} | Pearson {te_m['pearson']:.3f}")

    meta = {
        "model": f"CNN {args.arch} (ImageNet pretrained) -> Hb regresi",
        "crops_csv": str(args.crops), "n_images": len(df),
        "n_train": len(tr_df), "n_val": len(va_df), "n_test": len(te_df),
        "epochs_done": best_epoch, "patience": no_improve,
        "best_val_mae_gperL": best_mae, "best_val_mae_g_dl": round(best_mae / 10, 3),
        "test_mae_gperL": te_m["mae_gperL"], "test_mae_g_dl": round(te_m["mae_gperL"] / 10, 3),
        "test_rmse_gperL": te_m["rmse_gperL"], "test_r2": te_m["r2"],
        "test_pearson": te_m["pearson"], "target_unit": "g/L",
        "split": "per patient (uuid) 80/10/10, seed=42",
        "loss": "SmoothL1", "optimizer": "AdamW", "lr": args.lr,
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame({"PATIENT_UUID": te_df["PATIENT_UUID"], "MODALITY": te_df["MODALITY"],
                  "HB_LEVEL_GperL": te_y, "PRED_GperL": te_p}).to_csv(
        out_dir / "test_predictions.csv", index=False)
    print(f"saved: {out_dir}/best.pt + model_metadata.json + test_predictions.csv")


if __name__ == "__main__":
    sys.exit(main())