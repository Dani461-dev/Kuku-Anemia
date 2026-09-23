#!/usr/bin/env python3
"""C11 — CNN kuku tahan-dua-domain (kertas putih + background bebas).

Berbasis c6 + dua lever baru:
  1. --aug strong : augmentasi warna/lighting KUAT (ColorJitter 0.35, grayscale
     5%, GaussianBlur 10%, RandomErasing 25%) -> model belajar perbandingan
     relatif kuku-vs-kulit, bukan warna absolut kamera/cahaya.
  2. --mix-msu   : tambahkan 200 foto MSU-250 (background bebas) ke training
     (50 MSU lainnya di-holdout untuk evaluasi jujur). MSU-test ids disimpan
     di model_metadata.json -> c9 --exclude-mix-tag memakai info itu.

Split sewa tetap 80/10/10 seed 42 (identik C4/C6) -> MAE dalam-domain
dapat dibandingkan langsung. MSU hanya masuk bagian TRAIN (tidak val/test).

Usage:
    python experiments/hb_newdata/c11_train_cnn_robust.py [--mix-msu] [--tag resnet18_robust] \
        [--arch resnet18] [--epochs 25] [--colornorm whitep98] [--tta] [--device 0]
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
from PIL import Image
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXP = Path(__file__).resolve().parent
MSU_CSV = EXP / "outputs" / "crops_msu.csv"
MSU_TRAIN_N = 200           # dari 250 MSU, sisa 50 untuk eval holdout


class ColorNormalize:
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
        else:  # grayworld
            refs = np.array([arr[..., c].mean() for c in range(3)])
            refs = np.clip(refs, 30.0, None)
        out = np.clip(arr / refs[None, None, :] * 255.0, 0, 255)
        return Image.fromarray(out.astype(np.uint8))


class CropsDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["CROP_PATH"].tolist()
        self.y = df["HB_LEVEL_GperL"].astype(float).to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
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


def rebalance_weights(y: np.ndarray) -> np.ndarray:
    bins = np.array([30, 70, 100, 120, 201])
    cat = np.digitize(y, bins[1:-1])
    counts = np.bincount(cat, minlength=len(bins) - 1)
    w = np.array([1.0 / max(c, 1) for c in counts])
    w = w[cat]
    return w / w.mean()


def make_transforms(train: bool, colornorm: str, strong: bool):
    from torchvision import transforms
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    cn = ColorNormalize(colornorm)
    if not train:
        return transforms.Compose([
            cn, transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), norm,
        ])
    if strong:
        return transforms.Compose([
            cn,
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.35, contrast=0.35,
                                   saturation=0.30, hue=0.06),
            transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.10),
            transforms.ToTensor(), norm,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    return transforms.Compose([
        cn,
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15,
                               saturation=0.10, hue=0.02),
        transforms.ToTensor(), norm,
    ])


def metrics(y, pred):
    return {"mae_gperL": float(skm.mean_absolute_error(y, pred)),
            "rmse_gperL": float(np.sqrt(skm.mean_squared_error(y, pred))),
            "r2": float(skm.r2_score(y, pred)),
            "pearson": float(np.corrcoef(y, pred)[0, 1])}


def evaluate(model, loader, device, tta: bool):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).squeeze(1)
            if tta:
                out = 0.5 * (out + model(torch.flip(x, dims=[3])).squeeze(1))
            ys.append(y.numpy()); preds.append(out.float().cpu().numpy())
    ys = np.concatenate(ys); preds = np.concatenate(preds)
    return metrics(ys, preds), ys, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, default=EXP / "outputs" / "crops.csv")
    ap.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    ap.add_argument("--colornorm", choices=["none", "whitep98", "grayworld"], default="whitep98")
    ap.add_argument("--rebalance", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mix-msu", action="store_true")
    ap.add_argument("--tag", default="resnet18_robust")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.crops.exists():
        sys.exit(f"crops.csv tidak ada: {args.crops}")
    df = pd.read_csv(args.crops)
    df = df[df["CROP_PATH"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"sewa crop: {len(df)} | arch={args.arch} strong_aug=True mix_msu={args.mix_msu}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    tr_idx, va_idx, te_idx, _, _ = split_by_patient(df, args.seed, 0.10, 0.10)
    tr_df = df.loc[tr_idx].reset_index(drop=True)
    va_df = df.loc[va_idx].reset_index(drop=True)
    te_df = df.loc[te_idx].reset_index(drop=True)

    msu_train_ids, msu_test_ids = [], []
    if args.mix_msu:
        msu = pd.read_csv(MSU_CSV)
        ids = sorted(msu["PATIENT_ID"].astype(str).unique())
        msu_train_ids = ids[:MSU_TRAIN_N]
        msu_test_ids = ids[MSU_TRAIN_N:]
        msu_tr = msu[msu["PATIENT_ID"].astype(str).isin(msu_train_ids)].copy()
        msu_tr = msu_tr.rename(columns={"PATIENT_ID": "PATIENT_UUID"})
        tr_df = pd.concat([tr_df, msu_tr[["PATIENT_UUID", "MODALITY", "HB_LEVEL_GperL", "CROP_PATH"]]],
                          ignore_index=True)
        print(f"mix-msu: train MSU {len(msu_tr)} | holdout MSU {len(msu_test_ids)} pasien")

    print(f"split per pasien | train {len(tr_df)} | val {len(va_df)} | test {len(te_df)}")

    ds_tr = CropsDataset(tr_df, make_transforms(True, args.colornorm, strong=True))
    ds_va = CropsDataset(va_df, make_transforms(False, args.colornorm, strong=True))
    ds_te = CropsDataset(te_df, make_transforms(False, args.colornorm, strong=True))
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=0)

    model = make_model(args.arch).to(device)
    criterion = nn.SmoothL1Loss(reduction="none")
    weights = rebalance_weights(tr_df["HB_LEVEL_GperL"].astype(float).to_numpy()) \
        if args.rebalance else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           patience=3, factor=0.5)
    scaler = torch.amp.GradScaler("cuda")

    out_dir = EXP / "models" / "cnn" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best_mae, best_state, best_epoch, no_improve = 1e9, None, -1, 0

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_b = 0.0, 0
        for bi, (x, y) in enumerate(dl_tr):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                out = model(x).squeeze(1)
                loss = criterion(out, y)
                if weights is not None:
                    w = torch.from_numpy(weights[bi * args.batch:(bi + 1) * args.batch]).to(device)
                    if len(w) < len(loss):        # batch terakhir lebih pendek
                        w = torch.from_numpy(weights[:len(loss)]).to(device)
                    loss = (loss * w).mean()
                else:
                    loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach()) * len(y); n_b += len(y)
        val_m, _, _ = evaluate(model, dl_va, device, tta=args.tta)
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
    te_m, te_y, te_p = evaluate(model, dl_te, device, tta=args.tta)
    print("\n===== TEST sewa (per pasien, set terpisah) =====")
    print(f"MAE {te_m['mae_gperL']:.2f} g/L ({te_m['mae_gperL']/10:.2f} g/dL) | "
          f"R2 {te_m['r2']:+.3f} | Pearson {te_m['pearson']:.3f}")
    te_pdf = pd.DataFrame({"PATIENT_UUID": te_df["PATIENT_UUID"],
                           "HB_LEVEL_GperL": te_y, "PRED_GperL": te_p})
    pp = te_pdf.groupby("PATIENT_UUID").mean(numeric_only=False)
    pp_mae = float(np.abs(pp["HB_LEVEL_GperL"] - pp["PRED_GperL"]).mean())
    pp_r2 = float(skm.r2_score(pp["HB_LEVEL_GperL"], pp["PRED_GperL"]))
    print(f"per-patient (mean 2 foto): MAE {pp_mae:.2f} g/L ({pp_mae/10:.2f} g/dL) | R2 {pp_r2:+.3f}")

    meta = {
        "model": f"CNN {args.arch} robust (ImageNet pretrained) -> Hb regresi",
        "colornorm": args.colornorm, "rebalance": args.rebalance, "tta": args.tta,
        "strong_aug": True, "mix_msu": args.mix_msu,
        "msu_train_n": len(msu_train_ids), "msu_train_ids": msu_train_ids,
        "msu_test_ids": msu_test_ids,
        "crops_csv": str(args.crops), "n_images": len(df),
        "n_train": len(tr_df), "n_val": len(va_df), "n_test": len(te_df),
        "epochs_done": best_epoch, "best_val_mae_gperL": best_mae,
        "test_mae_gperL": te_m["mae_gperL"], "test_mae_g_dl": round(te_m["mae_gperL"] / 10, 3),
        "test_rmse_gperL": te_m["rmse_gperL"], "test_r2": te_m["r2"],
        "test_pearson": te_m["pearson"],
        "per_patient_mae_gperL": round(pp_mae, 3), "per_patient_mae_g_dl": round(pp_mae / 10, 3),
        "target_unit": "g/L",
        "split": "per patient (uuid) 80/10/10, seed=42 (sama dgn C4/C6)",
        "loss": "SmoothL1" + (" weighted" if args.rebalance else ""),
        "optimizer": "AdamW", "lr": args.lr,
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    te_pdf.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"saved: {out_dir}/best.pt + model_metadata.json + test_predictions.csv ({out_dir})")


if __name__ == "__main__":
    sys.exit(main())