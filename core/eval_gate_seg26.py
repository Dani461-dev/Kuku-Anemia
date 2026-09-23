#!/usr/bin/env python3
"""Gate eval fine-tuned seg26 vs baseline sebelum masuk runtime.

Gate 1 — V2 test (domain asli): Dice/IoU/detection harus tidak turun drastis
         (target Dice >= 0.80).
Gate 2 — Foto app (data/full_hand): 5/5 kuku terdeteksi + mean conf naik.
Gate 3 — MSU (data/photo): jumlah deteksi tidak regresi (sanity).

Usage:
    python3 core/eval_gate_seg26.py --new PATH/TO/weights/best.pt
                                   [--baseline .../runs/seg26/weights/best.pt]
                                   [--device 0] [--app-dir data/full_hand]
                                   [--v2-limit N] [--msu-limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = (PROJECT_ROOT / "experiments/yolo26_seg/runs/seg26/weights/best.pt")
V2 = PROJECT_ROOT / "Data Tambahan" / "archive (12)" / "NailSegmentationDatasetV2"
V2_IMG = V2 / "test" / "images"
V2_MSK = V2 / "test" / "masks"

sys.path.insert(0, str(PROJECT_ROOT))
from core.seg_detector import Yolo26SegDetector  # noqa: E402


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    s = float(a.sum()) + float(b.sum())
    return 2.0 * inter / s if s > 0 else 0.0


def load_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def eval_v2(det, limit: int) -> dict:
    imgs = sorted(V2_IMG.glob("*.jpg"))
    if limit:
        imgs = imgs[:limit]
    dices, ious, dets = [], [], []
    for p in imgs:
        img = load_rgb(p)
        gt = cv2.imread(str(V2_MSK / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE) > 0
        res = det.segment(img)
        dets.append(int(len(res.instances) > 0))
        if not res or not gt.any():
            dices.append(0.0 if gt.any() else 1.0)
            ious.append(0.0 if gt.any() else 1.0)
            continue
        pred = np.zeros_like(gt)
        for inst in res.instances:
            pred |= (inst.mask > 0)
        tp = float((pred & gt).sum())
        dices.append(dice(pred, gt))
        ious.append(tp / float((pred | gt).sum()) if (pred | gt).sum() else 0.0)
    return dict(images=len(imgs), dice=float(np.mean(dices)),
                iou=float(np.mean(ious)), det_rate=float(np.mean(dets)))


def eval_app(det, app_dir: Path) -> dict:
    imgs = sorted(p for p in app_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    counts, confs = [], []
    for p in imgs:
        res = det.segment(load_rgb(p))
        counts.append(len(res.instances))
        confs += [i.confidence for i in res.instances]
    return dict(photos=len(imgs), n_mean=float(np.mean(counts)),
                got5=sum(1 for c in counts if c == 5), conf_mean=float(np.mean(confs)))


def eval_msu(det, msu_dir: Path, limit: int) -> dict:
    imgs = sorted(p for p in msu_dir.glob("*.jpg"))
    if limit:
        imgs = imgs[:limit]
    counts = []
    for p in imgs:
        res = det.segment(load_rgb(p))
        counts.append(len(res.instances))
    return dict(photos=len(imgs), n_mean=float(np.mean(counts)),
                all_detected=int(np.all(np.array(counts) > 0)))


def run(weights: Path, device: str, two_pass: bool, app_dir: Path,
        v2_limit: int, msu_limit: int) -> dict:
    det = Yolo26SegDetector(weights=weights, conf=0.30, imgsz=640,
                            device=device, fallback=two_pass)
    return dict(v2=eval_v2(det, v2_limit),
                app=eval_app(det, app_dir),
                msu=eval_msu(det, PROJECT_ROOT / "data" / "photo", msu_limit))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True, help="weights hasil fine-tune")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--device", default="0")
    ap.add_argument("--app-dir", type=Path, default=PROJECT_ROOT / "data" / "full_hand")
    ap.add_argument("--v2-limit", type=int, default=0)
    ap.add_argument("--msu-limit", type=int, default=0)
    ap.add_argument("--no-two-pass", action="store_true",
                    help="evaluasi tanpa fallback (fair utk gate 1 & msu)")
    args = ap.parse_args()

    for w in (args.baseline, args.new):
        if not w.exists():
            sys.exit(f"weights tidak ada: {w}")

    two_pass = not args.no_two_pass
    base = run(args.baseline, args.device, two_pass, args.app_dir,
               args.v2_limit, args.msu_limit)
    new = run(args.new, args.device, two_pass, args.app_dir,
              args.v2_limit, args.msu_limit)

    two = f" dua-pass={two_pass}"
    print(f"GATE eval seg26 (conf 0.30 @ 640){two}")
    print("─" * 74)
    hdr = f"{'metrik':28s} {'baseline':>12s} {'fine-tune':>12s} {'selisih':>12s}  {'status'}"
    print(hdr)
    print("─" * 74)

    def row(name, b, n, ok=None):
        print(f"{name:28s} {b:>12.4f} {n:>12.4f} {(n - b):>+12.4f}  {ok}")

    for k in ("dice", "iou", "det_rate"):
        b, n = base["v2"][k], new["v2"][k]
        ok = "✅" if (k == "det_rate" and n >= b) or (k != "det_rate" and n >= b - 0.02) else "⚠️"
        row(f"V2 {k} ({base['v2']['images']} img)", b, n, ok)
    b, n = base["app"]["conf_mean"], new["app"]["conf_mean"]
    row(f"app mean conf ({base['app']['photos']} foto)", b, n,
        "✅" if n > b else "⚠️")
    b, n = base["app"]["got5"], new["app"]["got5"]
    print(f"{'app 5/5 foto':28s} {b:>12d} {n:>12d} {'':>12s}  {'✅' if n >= b else '⚠️'}")
    b, n = base["msu"]["n_mean"], new["msu"]["n_mean"]
    row(f"MSU deteksi mean ({base['msu']['photos']})", b, n,
        "✅" if n >= b - 0.5 else "⚠️")

    print("─" * 74)
    verdict = all([
        new["v2"]["dice"] >= 0.80,
        new["v2"]["dice"] >= base["v2"]["dice"] - 0.02,
        new["app"]["got5"] >= base["app"]["got5"],
        new["msu"]["n_mean"] >= base["msu"]["n_mean"] - 0.5,
    ])
    print("VERDICT:", "✅ LOLOS — boleh masuk runtime" if verdict
          else "⛔ GAGAL GATE — pertahankan dua-pass saat ini")


if __name__ == "__main__":
    sys.exit(main())