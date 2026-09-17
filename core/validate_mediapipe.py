#!/usr/bin/env python3
"""Validasi jalur runtime MediaPipe (T1/T2).

Untuk setiap foto (default: data/full_hand/):
  MediaPipe (3 jari, format metadata) → box kuku/kulit →
  mask jari tengah → white (chart/auto) → 42 fitur → prediksi Hb → kategori WHO

Hasil:
  - overlay visual  → core/outputs/viz/mediapipe_<pid>.jpg
  - ringkasan CSV   → data/output/predictions.csv  (format ala metadata)

Flag kewajaran (T2):
  - Hb dalam rentang fisiologis 4–18 g/dL
  - mask coverage 15–99%
  - confidence hand deteksi >= threshold
  - rentang fitur terhadap distribusi training (deteksi domain-shift)

Jalankan:
    python3 core/validate_mediapipe.py [--input-dir data/full_hand] [--gender female]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config as cfg  # noqa: E402
from core.categorize import categorize_hb  # noqa: E402
from core.detectors import DetectedFinger  # noqa: E402
from core.hand_landmarks import HandGeometry, HandLandmarkExtractor, derive_all_boxes  # noqa: E402
from core.inference import NailHbModel  # noqa: E402
from core.masking import best_nail_mask  # noqa: E402
from core.pipeline import crop_from_box, load_rgb, patient_features  # noqa: E402

FINGERS = [8, 12, 16]  # index, middle, ring
HB_MIN_OK, HB_MAX_OK = 4.0, 18.0        # g/dL
COV_MIN_OK, COV_MAX_OK = 0.15, 0.99
CONF_MIN_OK = 0.5
FRAC_FEAT_OK = 0.8                       # >= 80% fitur dalam rentang training


def ordered_metadata_boxes(derived):
    """Urutkan box vertikal seperti metadata: NAIL_1=terbawah, NAIL_2=tengah, NAIL_3=teratas."""
    ordered = sorted(derived, key=lambda b: (b.nail_box[0] + b.nail_box[2]) / 2, reverse=True)
    return ordered


def training_feature_bounds() -> dict[str, tuple[float, float]]:
    """Per-feature [p1, p99] dari fitur training (untuk deteksi domain-shift)."""
    candidates = sorted(cfg.OUTPUTS_DIR.glob("features_gt_fixed*.csv"))
    nomask = [p for p in candidates if p.stem.endswith("_nomask")]
    candidates = [p for p in candidates if p not in nomask]
    if not candidates:
        return {}
    df = pd.read_csv(candidates[-1])
    features = [c for c in df.columns
                if c.startswith(("NAIL_", "SKIN_")) and not c.startswith(("_",))]
    bounds = {}
    for c in features:
        col = df[c].astype(float)
        bounds[c] = (float(np.percentile(col.dropna(), 1)), float(np.percentile(col.dropna(), 99)))
    return bounds


def feature_plausibility(vector, order, bounds) -> tuple[bool, float]:
    """Fraksi fitur di dalam rentang training; warn kalau < FRAC_FEAT_OK."""
    if not bounds:
        return True, 1.0
    in_b = 0
    for name, value in zip(order, vector):
        lo, hi = bounds.get(name, (None, None))
        if lo is None:
            continue
        if lo <= value <= hi:
            in_b += 1
    frac = in_b / max(1, len([n for n in order if n in bounds]))
    return frac >= FRAC_FEAT_OK, frac


def draw_overlay(img_rgb, landmarks, nails, skins, middle_idx, middle_mask, score,
                 hb_g_dl, category, flags_ok, issues, pid, out_path):
    canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]

    if landmarks is not None:
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)  # yellow landmarks

    for i, (nail, skin) in enumerate(zip(nails, skins)):
        color = (0, 200, 0) if i == middle_idx else (0, 120, 0)
        t, l, b, r = nail
        cv2.rectangle(canvas, (l, t), (r, b), color, 2)
        cv2.putText(canvas, f"N{i+1}", (l, max(t - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        t, l, b, r = skin
        cv2.rectangle(canvas, (l, t), (r, b), (200, 120, 0), 1)
        cv2.putText(canvas, f"S{i+1}", (l, max(t - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 120, 0), 1, cv2.LINE_AA)

    if middle_mask is not None and middle_idx >= 0:
        nail = nails[middle_idx]
        t, l, b, r = nail
        reg = canvas[t:b, l:r]
        if reg.size:
            m = cv2.resize(middle_mask, (r - l, b - t), interpolation=cv2.INTER_NEAREST)
            reg[m > 0] = (255, 255, 255)  # highlight mask in white
            canvas[t:b, l:r] = cv2.addWeighted(reg, 0.6, canvas[t:b, l:r], 0.4, 0)

    status = "OK" if flags_ok else "; ".join(issues[:4]) if issues else "?  "
    cv2.putText(canvas, f"pid={pid} conf={score:.2f} hb={hb_g_dl if hb_g_dl else 'NA'} "
                f"({category if category else 'NA'}) flags={status}",
                (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"pid={pid} conf={score:.2f} hb={hb_g_dl if hb_g_dl else 'NA'} "
                f"({category if category else 'NA'}) flags={status}",
                (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def process_image(img_path: Path, extractor, model, bounds, gender: str,
                  white_source: str, conf_min: float, out_dir: Path) -> dict:
    pid = img_path.stem
    img = load_rgb(str(img_path))
    h, w = img.shape[:2]
    landmarks, score = extractor.detect_ranked(img)
    row = {
        "PATIENT_ID": pid, "IMAGE": img_path.name, "DETECTED": 0,
        "HAND_CONFIDENCE": round(score, 3), "N_FINGERS": 0,
        "NAIL_BOUNDING_BOXES": "[]", "SKIN_BOUNDING_BOXES": "[]",
        "MASK_COVERAGE": None, "PREDICTED_HB_GperL": None, "PREDICTED_HB_GdL": None,
        "CATEGORY": None, "GENDER": gender, "WHITE_SOURCE": white_source,
        "MASK_METHOD": "auto", "FLAGS_OK": 0, "ISSUES": "no hand detected",
    }

    nails = skins = []
    middle_idx = -1
    middle_mask = None
    hb = None
    category = None

    if landmarks is not None:
        derived = ordered_metadata_boxes(derive_all_boxes(landmarks, w, h, HandGeometry()))
        nails = [b.nail_box for b in derived]
        skins = [b.skin_box for b in derived]
        middle_idx = len(derived) // 2 if derived else -1
        row["N_FINGERS"] = len(derived)
        row["NAIL_BOUNDING_BOXES"] = json.dumps(nails)
        row["SKIN_BOUNDING_BOXES"] = json.dumps(skins)

        if middle_idx >= 0:
            mid = derived[middle_idx]
            boxes = DetectedFinger(finger_id=mid.finger_id, nail_box=mid.nail_box,
                                   skin_box=mid.skin_box, confidence=score, source="mediapipe")
            feats = patient_features(img, boxes, white_source=white_source, use_mask=True)
            coverage = float(feats.pop("_MASK_COVERAGE", 1.0))
            vector = np.array([feats.get(c, 0.0) for c in model.feature_order], dtype=float)
            hb_perl = float(model.model.predict(vector.reshape(1, -1))[0])
            hb = hb_perl / 10.0
            category = categorize_hb(hb, gender)
            row["MASK_COVERAGE"] = round(coverage, 3)
            row["PREDICTED_HB_GperL"] = round(hb_perl, 1)
            row["PREDICTED_HB_GdL"] = round(hb, 2)
            row["CATEGORY"] = category
            nail_crop = crop_from_box(img, mid.nail_box)
            middle_mask = best_nail_mask(nail_crop, method="auto")
    else:
        row["ISSUES"] = "no hand detected"

    # ── T2 flags ─────────────────────────────────────────────
    issues = []
    if landmarks is None:
        issues.append("no_hand_detected")
    elif score < conf_min:
        issues.append("low_confidence")
    if hb is not None and not (HB_MIN_OK <= hb <= HB_MAX_OK):
        issues.append(f"hb_out_of_range({hb:.1f})")
    if row["MASK_COVERAGE"] is not None and not (COV_MIN_OK <= row["MASK_COVERAGE"] <= COV_MAX_OK):
        issues.append(f"mask_coverage({row['MASK_COVERAGE']})")
    feat_ok, frac = True, 1.0
    if hb is not None:
        feat_ok, frac = feature_plausibility(vector, model.feature_order, bounds)
        if not feat_ok:
            issues.append(f"features_out_of_training_range({frac:.0%})")
    flags_ok = landmarks is not None and not issues
    row["FLAGS_OK"] = 1 if flags_ok else 0
    row["ISSUES"] = "; ".join(issues) if issues else ""

    if landmarks is not None:
        row["DETECTED"] = 1

    out_dir.mkdir(parents=True, exist_ok=True)
    draw_overlay(img, landmarks, nails, skins, middle_idx, middle_mask, score,
                 hb, category, flags_ok, issues, pid, out_dir / f"mediapipe_{pid}.jpg")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=cfg.PROJECT_ROOT / "data" / "full_hand")
    parser.add_argument("--out-csv", type=Path, default=cfg.PROJECT_ROOT / "data" / "output" / "predictions.csv")
    parser.add_argument("--gender", default="female", choices=["female", "male"])
    parser.add_argument("--white", default="chart", choices=["chart", "auto"])
    parser.add_argument("--conf-min", type=float, default=CONF_MIN_OK)
    parser.add_argument("--viz-dir", type=Path, default=cfg.OUTPUTS_DIR / "viz")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"[T1] Folder input belum ada: {args.input_dir}")
        print("     Taruh foto tangan penuh di sini (idealnya dengan kartu warna ArUco).")
        return 0

    imgs = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        print(f"[T1] Tidak ada foto di {args.input_dir}")
        return 0

    print(f"[T1] {len(imgs)} foto -> {args.input_dir}")
    extractor = HandLandmarkExtractor()
    model = NailHbModel()
    bounds = training_feature_bounds()

    rows = [process_image(p, extractor, model, bounds, args.gender, args.white,
                          args.conf_min, args.viz_dir) for p in imgs]
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[T1] Ringkasan CSV -> {args.out_csv}")
    print(df[["PATIENT_ID", "DETECTED", "PREDICTED_HB_GdL", "CATEGORY", "FLAGS_OK", "ISSUES"]].to_string(index=False))
    print(f"[T1] Overlay -> {args.viz_dir.resolve()}")


if __name__ == "__main__":
    sys.exit(main())