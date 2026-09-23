#!/usr/bin/env python3
"""Pseudo-label app-domain photos -> YOLO-seg polygons for seg26 fine-tune.

Sumber  : folder foto domain aplikasi (default data/full_hand).
Deteksi : Yolo26SegDetector dua-pass (persis jalur runtime app).
Label   : mask seg26 dengan conf >= --min-conf dipertahankan sebagai GT awal.
QA otomatis per foto:
    - jumlah kuku  >= --min-nails
    - box tiap kuku >= 12px tiap sisi
    - mask fill (area mask / area box) dalam [0.30, 0.95]  (hindari speckle)
    - non-overlap antar kuku (dedupe IoU sudah di detektor; dicek ulang)
Output  : experiments/yolo26_seg/dataset_app/{images,labels}/{train,val}
          + overlay QA (dataset_app/qa/*.jpg) utk cek visual
          + data_app.yaml (V2 + app) utk Ultralytics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJ = Path(__file__).resolve().parent                      # experiments/yolo26_seg
SEG26_BEST = PROJ / "runs" / "seg26" / "weights" / "best.pt"
V2_DATASET = PROJ / "dataset"

sys.path.insert(0, str(PROJECT_ROOT))
from core.seg_detector import Yolo26SegDetector  # noqa: E402

MIN_BOX_SIDE = 12
FILL_RANGE = (0.30, 0.95)


def mask_to_polygons(mask: np.ndarray, img_w: int, img_h: int) -> list[list[float]]:
    """Salin logika prepare_seg26: mask biner -> poligon YOLO ternormalisasi."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img_w * img_h
    polys: list[list[float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1e-4 * img_area or len(cnt) < 3:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.004 * peri, True).reshape(-1, 2).astype(float)
        if len(approx) > 100:
            idx = np.linspace(0, len(approx) - 1, 100).astype(int)
            approx = approx[idx]
        approx[:, 0] /= img_w
        approx[:, 1] /= img_h
        approx = np.clip(approx, 0.0, 1.0)
        polys.append(approx.ravel().tolist())
    return polys


def box_iou(a: list[int], b: list[int]) -> float:
    it = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it)
    return it / ua if ua > 0 else 0.0


def qa_ok(insts: list, min_nails: int) -> tuple[bool, str]:
    """Return (lolos, alasan). insts: NailInstance setelah filter conf."""
    if len(insts) < min_nails:
        return False, f"hanya {len(insts)} kuku (< {min_nails})"
    for i, inst in enumerate(insts):
        t, l, b, r = inst.nail_box
        if min(b - t, r - l) < MIN_BOX_SIDE:
            return False, f"kuku #{i} terlalu kecil ({b - t}x{r - l}px)"
        box_area = max(1, (b - t) * (r - l))
        fill = float(inst.mask[t:b, l:r].sum()) / box_area
        if not (FILL_RANGE[0] <= fill <= FILL_RANGE[1]):
            return False, f"kuku #{i} mask fill {fill:.2f} di luar {FILL_RANGE}"
    for i in range(len(insts)):
        for j in range(i + 1, len(insts)):
            if box_iou(insts[i].nail_box, insts[j].nail_box) > 0.5:
                return False, f"kuku #{i} & #{j} overlap (IoU {box_iou(insts[i].nail_box, insts[j].nail_box):.2f})"
    return True, ""


def save_overlay(img: np.ndarray, insts: list, path: Path, note: str) -> None:
    ov = img.copy()
    for i, inst in enumerate(insts):
        col = (50, 200, 50) if inst.confidence >= 0.5 else (0, 0, 255)
        t, l, b, r = inst.nail_box
        cv2.rectangle(ov, (l, t), (r, b), col, 3)
        cv2.putText(ov, f"{inst.confidence:.2f}", (l, max(0, t - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(ov, note, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "full_hand")
    ap.add_argument("--tag", default="app")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-conf", type=float, default=0.50,
                    help="simpan mask dgn confidence >= ini sebagai label")
    ap.add_argument("--min-nails", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="fraksi foto utk val (0 = semua train; gate dieval manual)")
    ap.add_argument("--out", type=Path, default=PROJ / "dataset_app")
    ap.add_argument("--no-yaml", action="store_true")
    args = ap.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"input dir tak ada: {args.input_dir}")
    imgs = sorted(p for p in args.input_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        sys.exit(f"tidak ada gambar di {args.input_dir}")
    if not SEG26_BEST.exists():
        sys.exit(f"seg26 best tidak ada: {SEG26_BEST}")

    det = Yolo26SegDetector(weights=SEG26_BEST, device=args.device, fallback=True)
    out = args.out
    qa_dir = out / "qa"

    n_acc = n_rej = 0
    n_val = max(1, int(round(len(imgs) * args.val_frac))) if args.val_frac > 0 else 0
    for idx, f in enumerate(imgs):
        img = cv2.imread(str(f))
        if img is None:
            print(f"  SKIP read-gagal: {f.name}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = det.segment(img)
        insts = sorted((i for i in seg.instances if i.confidence >= args.min_conf),
                       key=lambda i: i.confidence, reverse=True)[:5]
        ok, reason = qa_ok(insts, args.min_nails)
        if not ok:
            n_rej += 1
            save_overlay(img, insts, qa_dir / f"{idx:04d}_{f.stem[:30]}_REJ.jpg", f"REJ: {reason}")
            print(f"  REJECT {f.name[:44]:46s} {reason}")
            continue
        split = "val" if (n_val and idx < n_val) else "train"
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{args.tag}_{idx:04d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        lines = []
        for inst in insts:
            for poly in mask_to_polygons(inst.mask, img.shape[1], img.shape[0]):
                lines.append("0 " + " ".join(f"{v:.6f}" for v in poly))
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        save_overlay(img, insts, qa_dir / f"{idx:04d}_{f.stem[:30]}.jpg",
                     f"OK {split} ({len(insts)} kuku)")
        n_acc += 1
        print(f"  ACCEPT {f.name[:44]:46s} -> {split}/{stem}.jpg ({len(insts)} kuku)")

    print(f"\ndiproses {len(imgs)} foto | accepted {n_acc} | rejected {n_rej}")
    print(f"dataset_app: {out}")

    if not args.no_yaml and n_acc:
        train_dirs, val_dirs = [], []
        for d in ("dataset/images/train", f"dataset_app/images/train"):
            if (PROJ / d).exists() and any((PROJ / d).glob("*")):
                train_dirs.append(d)
        for d in ("dataset/images/val", f"dataset_app/images/val"):
            if (PROJ / d).exists() and any((PROJ / d).glob("*")):
                val_dirs.append(d)
        yaml_txt = ["path: " + PROJ.as_posix(), "train:", *[f"  - {d}" for d in train_dirs],
                    "val:", *[f"  - {d}" for d in val_dirs], "names:", "  0: nail"]
        yaml_path = PROJ / "data_app.yaml"
        yaml_path.write_text("\n".join(yaml_txt) + "\n")
        print(f"data_app.yaml -> {yaml_path}")
        print("\n".join(yaml_txt))


if __name__ == "__main__":
    sys.exit(main())