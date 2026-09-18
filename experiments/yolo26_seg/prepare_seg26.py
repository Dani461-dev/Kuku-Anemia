#!/usr/bin/env python3
"""Convert NailSegmentationDatasetV2 masks into YOLO-seg polygon labels.

The V2 dataset ships binary masks (0/255) — one mask per image that may
contain several disconnected nail regions (≈5 nails/finger). Each connected
component becomes a separate YOLO-seg instance (class 0 = nail).

Output (Ultralytics YOLO-seg layout):
    experiments/yolo26_seg/dataset/
    ├── images/{train,val,test}/*.jpg
    ├── labels/{train,val,test}/*.txt      # '0 x1 y1 x2 y2 ...' (normalised)
    └── data.yaml

Usage:
    python3 experiments/yolo26_seg/prepare_seg26.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_V2 = PROJECT_ROOT / "Data Tambahan" / "archive (12)" / "NailSegmentationDatasetV2"
OUT_ROOT = Path(__file__).resolve().parent / "dataset"

MIN_AREA_RATIO = 1e-4     # drop components smaller than 0.01% of image
MAX_POINTS = 100          # ultralytics segment label cap
EPSILON_FRAC = 0.004      # approxPolyDP epsilon as fraction of contour perimeter


def mask_to_polygons(mask: np.ndarray, img_w: int, img_h: int) -> list[list[float]]:
    """Return a list of polygon point-lists (normalised [x1,y1,x2,y2,...])."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img_w * img_h
    polys: list[list[float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_RATIO * img_area or len(cnt) < 3:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_FRAC * peri, True).reshape(-1, 2).astype(float)
        # scale down to at most MAX_POINTS (keep evenly spaced)
        if len(approx) > MAX_POINTS:
            idx = np.linspace(0, len(approx) - 1, MAX_POINTS).astype(int)
            approx = approx[idx]
        approx[:, 0] /= img_w
        approx[:, 1] /= img_h
        approx = np.clip(approx, 0.0, 1.0)
        polys.append(approx.ravel().tolist())
    return polys


def build_split(split: str, args) -> tuple[int, int]:
    src_img = DATASET_V2 / split / "images"
    src_msk = DATASET_V2 / split / "masks"
    dst_img = OUT_ROOT / "images" / split
    dst_lbl = OUT_ROOT / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n_img, n_lbl = 0, 0
    problems = []
    for mpath in sorted(src_msk.glob("*.png")):
        img_path = src_img / (mpath.stem + ".jpg")
        if not img_path.exists():
            problems.append(f"missing image for {mpath.name}")
            continue

        out_name = mpath.stem + ".txt"
        dl = dst_img / (mpath.stem + ".jpg")
        label_done = (dst_lbl / out_name).exists()
        if args.copy:
            img_done = dl.is_file() and not dl.is_symlink()
        else:
            img_done = dl.exists() or dl.is_symlink()
        if label_done and img_done:
            n_poly = len((dst_lbl / out_name).read_text().splitlines())
            n_img += 1
            n_lbl += n_poly
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            problems.append(f"unreadable image {img_path.name}")
            continue
        h, w = img.shape[:2]
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            problems.append(f"unreadable mask {mpath.name}")
            continue

        polys = mask_to_polygons((mask > 0).astype(np.uint8), w, h)
        if not polys:
            problems.append(f"no polygons in {mpath.name}")
            continue

        lines = ["0 " + " ".join(f"{v:.6f}" for v in poly) for poly in polys]
        (dst_lbl / out_name).write_text("\n".join(lines) + "\n")

        if args.copy:
            if dl.is_symlink() or (dl.exists() and not dl.is_file()):
                dl.unlink()
                dl = dst_img / (mpath.stem + ".jpg")
            shutil.copy2(img_path, dl)
        else:
            if dl.exists() or dl.is_symlink():
                dl.unlink()
            dl.symlink_to(img_path)

        n_img += 1
        n_lbl += len(polys)

    if problems and not args.quiet:
        for p in problems[:5]:
            print(f"  WARN {split}: {p}")
    return n_img, n_lbl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    parser.add_argument("--quiet", action="store_true", help="suppress warnings")
    args = parser.parse_args()

    if not DATASET_V2.exists():
        sys.exit(f"V2 dataset not found: {DATASET_V2}")

    total = {"images": 0, "instances": 0}
    for split in ("train", "val", "test"):
        n_img, n_lbl = build_split(split, args)
        total["images"] += n_img
        total["instances"] += n_lbl
        print(f"  {split}: {n_img} images, {n_lbl} nail instances")

    data_yaml = (
        f"path: {OUT_ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: nail\n"
    )
    (OUT_ROOT / "data.yaml").write_text(data_yaml)
    print(f"  total: {total['images']} images, {total['instances']} instances")
    print(f"  data.yaml -> {OUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    sys.exit(main())