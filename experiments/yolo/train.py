#!/usr/bin/env python3
"""Train YOLO11n detector (classes: NAIL, SKIN) on the prepared dataset.

Experimental / optional. The main pipeline does NOT depend on this.

Usage:
    python3 experiments/yolo/train.py [--epochs 60] [--imgsz 640] [--quick]

Prepare the dataset first (see prepare_yolo.py). Output weights:
    experiments/yolo/runs/detector/weights/best.pt
(consumed by core/detectors.YOLODetector).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "weights" / "yolo11n.pt"))
    parser.add_argument("--quick", action="store_true", help="3 epochs sanity run")
    args = parser.parse_args()

    data_yaml = PROJECT_ROOT / "experiments" / "yolo" / "dataset" / "data.yaml"
    runs_dir = PROJECT_ROOT / "experiments" / "yolo"
    if not data_yaml.exists():
        sys.exit(f"Missing {data_yaml}. Run experiments/yolo/prepare_yolo.py first.")

    from ultralytics import YOLO

    model = YOLO(args.model)
    epochs = 3 if args.quick else args.epochs

    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(runs_dir / "runs"),
        name="detector",
        seed=42,
        verbose=True,
    )
    print(f"Training finished in {(time.time() - t0) / 60:.1f} min")

    metrics = model.val(project=str(runs_dir / "runs"), name="val")
    print(f"val mAP50: {metrics.box.map50:.4f}")
    print(f"val mAP50-95: {metrics.box.map:.4f}")

    best = runs_dir / "runs" / "detector" / "weights" / "best.pt"
    print(f"Best weights: {best}")


if __name__ == "__main__":
    sys.exit(main())