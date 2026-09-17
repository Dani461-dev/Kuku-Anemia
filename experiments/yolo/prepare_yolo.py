#!/usr/bin/env python3
"""Convert dataset bounding boxes (from metadata.csv) into YOLO label format.

metadata.csv bounding boxes are stored as [top, left, bottom, right] in pixels
(0-based), one list per finger. Classes: 0=NAIL, 1=SKIN.

Output structure (Ultralytics YOLO layout):
    experiments/yolo/dataset/
    ├── images/train/*.jpg
    ├── images/val/*.jpg
    ├── labels/train/*.txt
    ├── labels/val/*.txt
    └── data.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_PHOTO_DIR = PROJECT_ROOT / "data" / "photo"
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "yolo" / "dataset"

CLASS_NAMES = {0: "NAIL", 1: "SKIN"}


def parse_bboxes(raw: str) -> list[list[int]]:
    boxes = json.loads(raw)
    return [[int(v) for v in box] for box in boxes]


def to_yolo(box, w: int, h: int) -> str:
    """Convert [top, left, bottom, right] pixel box to YOLO normalized
    'x_center y_center width height'."""
    top, left, bottom, right = box
    x_c = ((left + right) / 2.0) / w
    y_c = ((top + bottom) / 2.0) / h
    w_n = (right - left) / w
    h_n = (bottom - top) / h
    x_c = np.clip(x_c, 0.0, 1.0)
    y_c = np.clip(y_c, 0.0, 1.0)
    w_n = np.clip(w_n, 1.0 / w, 1.0)
    h_n = np.clip(h_n, 1.0 / h, 1.0)
    return f"{x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from metadata.csv")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--photos", type=Path, default=DEFAULT_PHOTO_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlink")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    print(f"Loaded {len(metadata)} patients from {args.metadata}")

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (args.out / sub / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    patient_ids = metadata["PATIENT_ID"].astype(int).values
    shuffled = patient_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_ids = set(shuffled[:n_val].tolist())

    stats = {"images": 0, "boxes": 0, "missing": [], "skipped": 0}
    for split, ids in (("train", patient_ids), ("val", patient_ids)):
        for pid in ids:
            assign = "val" if split == "val" and pid in val_ids else (
                "train" if split == "train" and pid not in val_ids else None
            )
            if assign is None:
                continue
            img_path = args.photos / f"{pid}.jpg"
            if not img_path.exists():
                stats["missing"].append(pid)
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                stats["skipped"] += 1
                continue
            h, w = img.shape[:2]

            row = metadata[metadata["PATIENT_ID"] == pid].iloc[0]
            lines = []
            for cls_id, col in ((0, "NAIL_BOUNDING_BOXES"), (1, "SKIN_BOUNDING_BOXES")):
                for box in parse_bboxes(row[col]):
                    lines.append(f"{cls_id} {to_yolo(box, w, h)}")
                    stats["boxes"] += 1

            out_img = args.out / "images" / assign / f"{pid}.jpg"
            out_lbl = args.out / "labels" / assign / f"{pid}.txt"
            if args.copy:
                shutil.copy2(img_path, out_img)
            else:
                if out_img.exists() or out_img.is_symlink():
                    out_img.unlink()
                out_img.symlink_to(img_path)
            out_lbl.write_text("\n".join(lines) + "\n")
            stats["images"] += 1

    for split in ("train", "val"):
        n = len(list((args.out / "images" / split).glob("*.jpg")))
        print(f"  {split}: {n} images")

    print(f"Total boxes written: {stats['boxes']}")
    if stats["missing"]:
        print(f"WARNING: images not found for patient ids: {stats['missing']}")
    if stats["skipped"]:
        print(f"WARNING: unreadable images skipped: {stats['skipped']}")

    data_yaml = (
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAMES[0]}\n"
        f"  1: {CLASS_NAMES[1]}\n"
    )
    (args.out / "data.yaml").write_text(data_yaml)
    print(f"data.yaml written to {args.out / 'data.yaml'}")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())