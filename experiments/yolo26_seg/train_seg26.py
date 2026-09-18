#!/usr/bin/env python3
"""Train YOLO26n-seg (nail instance segmentation) on NailSegmentationDatasetV2.

Pipeline:
    prepare_seg26.py  ->  dataset (images + polygon labels + data.yaml)
    train_seg26.py    ->  YOLO26n-seg weights (best.pt) + metrics

Usage:
    # sanity (pipeline check):
    python3 experiments/yolo26_seg/train_seg26.py --quick
    # full:
    python3 experiments/yolo26_seg/train_seg26.py --epochs 60 --imgsz 640

Output:
    experiments/yolo26_seg/runs/seg26/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = Path(__file__).resolve().parent / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60, help="training epochs (full run)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    default_device = "0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", type=str, default="yolo26n-seg.pt",
                        help="base weights; auto-downloaded if not local")
    parser.add_argument("--data", type=Path, default=None,
                        help="override data.yaml (e.g. a quick local subset)")
    parser.add_argument("--quick", action="store_true", help="1-epoch sanity run")
    parser.add_argument("--patience", type=int, default=30, help="early stopping")
    args = parser.parse_args()

    data_yaml = args.data or DATASET / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"Missing {data_yaml}. Run prepare_seg26.py first.")

    from ultralytics import YOLO

    model = YOLO(args.model)
    epochs = 1 if args.quick else args.epochs

    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(Path(__file__).parent / "runs"),
        name="seg26",
        seed=42,
        verbose=True,
        plots=True,
    )
    print(f"Training finished in {(time.time() - t0) / 60:.1f} min")

    if not args.quick:
        model_path = Path(__file__).parent / "runs" / "seg26"
        results = model.val(project=str(model_path.parent), name="val_seg26")
        print(f"val mAP50: {results.box.map50:.4f}")
        print(f"val mAP50-95: {results.box.map:.4f}")
        print(f"mask mAP50: {results.seg.map50:.4f}")
        print(f"mask mAP50-95: {results.seg.map:.4f}")


if __name__ == "__main__":
    sys.exit(main())