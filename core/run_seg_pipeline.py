#!/usr/bin/env python3
"""Runtime pipeline harness: YOLO26-seg -> nail masks -> boxes -> features -> Hb.

For each photo:
  1. YOLO26-seg detects nail instances (per-pixel masks).
  2. Middle finger (vertical median) is selected.
  3. Nail mask + derived skin band feed the 42 normalised features (seg masks).
  4. Hb + WHO category are predicted by the canonical Hb model.

Output:
  - overlay visuals   -> core/outputs/viz_seg/seg_<pid>.jpg
  - summary CSV       -> data/output/seg_predictions.csv

Usage:
    python3 core/run_seg_pipeline.py [--input data/full_hand] [--gender female]
                                     [--white chart] [--limit 20]
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config as cfg  # noqa: E402
from core.categorize import categorize_hb  # noqa: E402
from core.guard import assess_hb, feature_bounds  # noqa: E402
from core.pipeline import crop_from_box, load_rgb  # noqa: E402
from core.seg_detector import Yolo26SegDetector, nail_skin_masks  # noqa: E402
from core.inference import NailHbModel  # noqa: E402

DEFAULT_INPUT = cfg.PROJECT_ROOT / "data" / "full_hand"
DEFAULT_OUT_CSV = cfg.PROJECT_ROOT / "data" / "output" / "seg_predictions.csv"
DEFAULT_VIZ = cfg.OUTPUTS_DIR / "viz_seg"


def draw_overlay(img_rgb, result, mid_idx: int, nail_mask, skin_mask, out_path: Path):
    canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]

    for i, inst in enumerate(result.instances):
        t, l, b, r = inst.nail_box
        color = (0, 220, 0) if i == mid_idx else (0, 120, 0)
        cv2.rectangle(canvas, (l, t), (r, b), color, 2)
        cv2.putText(canvas, f"N{i + 1} {inst.confidence:.2f}", (l, max(t - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        # skin box
        skin = inst.skin_box(h, w)
        st, sl, sb, sr = skin
        cv2.rectangle(canvas, (sl, st), (sr, sb), (0, 140, 255), 1)
        cv2.putText(canvas, f"S{i + 1}", (sl, max(st - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1, cv2.LINE_AA)

    if nail_mask is not None:
        overlay = np.zeros_like(canvas)
        overlay[nail_mask > 0] = (0, 255, 0)
        overlay[skin_mask > 0] = (255, 0, 0)
        canvas = cv2.addWeighted(canvas, 0.85, overlay, 0.4, 0)

    cv2.imwrite(str(out_path), canvas)


def process_image(img_path: Path, detector, model, gender: str, white: str,
                  bounds: dict) -> dict:
    pid = img_path.stem
    img = load_rgb(str(img_path))
    result = detector.segment(img)
    row = {
        "PATIENT_ID": pid, "IMAGE": img_path.name, "SEGMENTED": 0,
        "N_INSTANCES": len(result.instances), "MASK_COVERAGE": None,
        "PREDICTED_HB_GdL": None, "CATEGORY": None, "FLAGS_OK": 0,
        "FEAT_FRAC": None, "ISSUES": "",
    }

    if not result:
        row["ISSUES"] = "no nail detected"
        return row

    # middle finger = instance whose nail-box vertical centre is the median
    ordered = sorted(result.instances,
                     key=lambda i: (i.nail_box[0] + i.nail_box[2]) / 2)
    mid_inst = ordered[len(ordered) // 2]
    mid_idx = next(i for i, inst in enumerate(result.instances) if inst is mid_inst)
    mid = mid_inst.to_detected_finger(img.shape[0], img.shape[1])
    nail_mask, skin_mask = nail_skin_masks(
        mid_inst, img.shape[0], img.shape[1])
    # nail-box coverage (match training definition of mask coverage)
    t, l, b, r = mid_inst.nail_box
    coverage = float((nail_mask[t:b, l:r] > 0).mean()) if (b > t and r > l) else 0.0

    hb = model.predict_masks(img, mid, nail_mask=nail_mask, skin_mask=skin_mask,
                             gender=gender, white_source=white)
    vector = model.features_for_masks(img, mid, nail_mask=nail_mask,
                                      skin_mask=skin_mask, white_source=white)
    row["SEGMENTED"] = 1
    row["MASK_COVERAGE"] = round(coverage, 4)
    row["PREDICTED_HB_GdL"] = hb["estimated_hb_g_dl"]
    row["CATEGORY"] = hb["category"]

    # plausibility guard -> reject (retake) instead of a wild prediction
    guard = assess_hb(coverage, hb["estimated_hb_g_dl"], vector[0],
                      model.feature_order, bounds, n_instances=len(result.instances))
    row["FEAT_FRAC"] = round(guard.feat_frac, 3)
    if not guard.ok:
        row["FLAGS_OK"] = 0
        row["ISSUES"] = guard.message
        row["PREDICTED_HB_GdL"] = None
        row["CATEGORY"] = None
    else:
        row["FLAGS_OK"] = 1

    out_viz = cfg.OUTPUTS_DIR / "viz_seg"
    out_viz.mkdir(parents=True, exist_ok=True)
    draw_overlay(img, result, mid_idx, nail_mask, skin_mask,
                 out_viz / f"seg_{pid}.jpg")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--gender", default="female", choices=["female", "male"])
    parser.add_argument("--white", default=None, choices=["chart", "auto", "fixed"],
                        help="white source (default: locked from the deployed Hb model metadata)")
    parser.add_argument("--weights", type=str, default=None,
                        help="override seg26 weights path")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="detection confidence threshold (pass utama, skala training)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO input size pass utama (default 640 = skala training)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="nonaktifkan dua-pass (fallback conf 0.10 @ imgsz 1280 utk foto susah)")
    parser.add_argument("--limit", type=int, default=0, help="limit number of photos")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"input dir not found: {args.input_dir}")
        return 1

    imgs = sorted(p for p in args.input_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit > 0:
        imgs = imgs[: args.limit]
    if not imgs:
        print(f"no images in {args.input_dir}")
        return 0

    print(f"photos: {len(imgs)} from {args.input_dir}")
    print(f"seg26 device: {args.device}")

    detector = Yolo26SegDetector(weights=args.weights, device=args.device,
                                 conf=args.conf, imgsz=args.imgsz,
                                 fallback=not args.no_fallback)
    model = NailHbModel()

    # white-source lock: prefer explicit CLI, else the model's training source
    white = args.white or model.meta.get("white_source", "auto")
    if args.white and args.white != model.meta.get("white_source", "auto"):
        print(f"WARNING: --white={args.white} != model white_source="
              f"{model.meta.get('white_source', 'auto')} (fitur bisa bergeser)")
    print(f"white source: {white} (locked from model metadata "
          f"if not passed) | model: {model.meta.get('features_csv')}")

    bounds = feature_bounds()
    rows = [process_image(p, detector, model, args.gender, white, bounds) for p in imgs]
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"summary -> {args.out_csv}")
    cols = ["PATIENT_ID", "SEGMENTED", "N_INSTANCES", "PREDICTED_HB_GdL",
            "CATEGORY", "FLAGS_OK", "ISSUES"]
    print(df[cols].to_string(index=False))
    detected = int(df["SEGMENTED"].sum())
    ok = int(df["FLAGS_OK"].sum()) if "FLAGS_OK" in df else detected
    print(f"\ndetected: {detected}/{len(df)} photos | guard OK: {ok}/{len(df)}")
    print(f"overlays -> {cfg.OUTPUTS_DIR / 'viz_seg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())