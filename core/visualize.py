#!/usr/bin/env python3
"""Visualisasi hasil ekstraksi fitur untuk beberapa sampel.

Menghasilkan satu gambar komposit per PID ke core/outputs/viz/ :
  Panel A: foto penuh + semua box (NAIL 3 jari, SKIN 3 jari) + region white reference
  Panel B: crop kuku jari tengah + overlay mask (kmeans / otsu)
  Panel C: fitur persentil RGB (bar chart ringkas)

Jalankan:
    python3 core/visualize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config as cfg
from core.pipeline import load_rgb, select_middle_finger, white_median_for, crop_from_box
from core.detectors import GTDetector
from core.features import calculate_features, feature_names, normalize_by_white
from core.masking import best_nail_mask

VIZ_DIR = cfg.OUTPUTS_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────
NAIL_COLORS = [(0, 200, 0), (0, 255, 80), (80, 255, 80)]   # green shades
SKIN_COLORS = [(200, 120, 0), (255, 160, 40), (255, 200, 80)]  # blue shades
WHITE_RECT_COLOR = (0, 0, 255)  # red in BGR
FINGER_LABELS = ["1 (index)", "2 (middle)", "3 (ring)"]
FILL_ALPHA = 0.25


def draw_box(img, box, color, label=None, thickness=2):
    """Draw a labelled rectangle on img (BGR)."""
    top, left, bot, right = box
    cv2.rectangle(img, (left, top), (right, bot), color, thickness)
    if label:
        cv2.putText(img, label, (left, max(top - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


def draw_mask_overlay(crop_rgb, mask):
    """Semi-transparent red overlay on masked pixels."""
    overlay = crop_rgb.copy()
    overlay[mask > 0] = (255, 100, 100)  # light-red tint
    return cv2.addWeighted(crop_rgb, 0.55, overlay, 0.45, 0)


def draw_percentile_bars(canvas, features, y_start, x_start, w, h, title=""):
    """Simple horizontal bar chart of 7 percentiles per channel in canvas."""
    colors_bgr = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # R, G, B
    ch_names = ["R", "G", "B"]
    levels = [5, 15, 25, 50, 75, 85, 95]
    bar_h = max(8, h // 28)
    y = y_start
    if title:
        cv2.putText(canvas, title, (x_start, y + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += bar_h + 4
    for ch_i, ch in enumerate(ch_names):
        vals = [features.get(f"SKIN_{ch}_p={p}", 0.0) for p in levels] + \
               [features.get(f"NAIL_{ch}_p={p}", 0.0) for p in levels]
        max_val = max(vals) if max(vals) > 0 else 1.0
        label_y = y + bar_h - 2
        cv2.putText(canvas, f"{ch}", (x_start, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors_bgr[ch_i], 1, cv2.LINE_AA)
        bx = x_start + 16
        for v in vals:
            bw = int((v / max_val) * (w - 20))
            cv2.rectangle(canvas, (bx, y), (bx + bw, y + bar_h - 2), colors_bgr[ch_i], -1)
            bx += bw + 2
        y += bar_h + 2
    return y


def build_composite(pid, meta_row, img_rgb, detected, out_path):
    """Build and save a multi-panel visualization for one patient."""
    nails = json.loads(meta_row["NAIL_BOUNDING_BOXES"])
    skins = json.loads(meta_row["SKIN_BOUNDING_BOXES"])
    hb = float(meta_row["HB_LEVEL_GperL"])

    # ── Panel A: full photo with all boxes ────────────────────────
    panel_a = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = panel_a.shape[:2]

    for i, (nail, skin) in enumerate(zip(nails, skins)):
        draw_box(panel_a, nail, NAIL_COLORS[i], f"NAIL_{i+1}", thickness=2)
        draw_box(panel_a, skin, SKIN_COLORS[i], f"SKIN_{i+1}", thickness=1)

    # white reference region
    wr = (cfg.WHITE_COLS[0], cfg.WHITE_ROWS[0], cfg.WHITE_COLS[1], cfg.WHITE_ROWS[1])
    top, left, bot, right = wr
    cv2.rectangle(panel_a, (left, top), (right, bot), WHITE_RECT_COLOR, 2)
    cv2.putText(panel_a, "WHITE", (left, top - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE_RECT_COLOR, 1, cv2.LINE_AA)

    # title
    cv2.putText(panel_a, f"PID {pid}  |  Hb = {hb:.0f} g/L = {hb/10:.1f} g/dL",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel_a, f"PID {pid}  |  Hb = {hb:.0f} g/L = {hb/10:.1f} g/dL",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # legend
    ly = h - 60
    for i, label in enumerate(FINGER_LABELS):
        cv2.putText(panel_a, label, (w - 130, ly + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, NAIL_COLORS[i], 1, cv2.LINE_AA)

    # ── Panel B: nail crop + mask overlay ──────────────────────────
    middle = select_middle_finger(detected)
    if middle is not None:
        nail_crop = crop_from_box(img_rgb, middle.nail_box)
        skin_crop = crop_from_box(img_rgb, middle.skin_box)
        wm = white_median_for(img_rgb, "fixed")
        mask_kmeans = best_nail_mask(nail_crop, method="kmeans")
        mask_otsu = best_nail_mask(nail_crop, method="otsu")

        nch = nail_crop.shape[0]
        ncw = nail_crop.shape[1]

        # nail original
        nail_bgr_orig = cv2.cvtColor(nail_crop, cv2.COLOR_RGB2BGR)
        # masked
        masked = draw_mask_overlay(nail_bgr_orig.copy(), mask_kmeans) if mask_kmeans is not None else nail_bgr_orig.copy()
        # otsu mask
        otsu_overlay = draw_mask_overlay(nail_bgr_orig.copy(), mask_otsu) if mask_otsu is not None else nail_bgr_orig.copy()

        # scale up for visibility
        scale = max(1, min(3, 180 // max(nch, 1)))
        big = lambda im: cv2.resize(im, (ncw * scale, nch * scale), interpolation=cv2.INTER_NEAREST)

        labels_panel_b = ["Nail crop (asli)", "Nail + K-means mask", "Nail + Otsu mask"]
        crops_b = [big(nail_bgr_orig), big(masked), big(otsu_overlay)]
        # resize all to same size (largest)
        max_h = max(c.shape[0] for c in crops_b)
        max_w = max(c.shape[1] for c in crops_b)
        canvas_b = np.zeros((max_h, max_w * 3 + 8, 3), dtype=np.uint8)
        for idx, (c, lbl) in enumerate(zip(crops_b, labels_panel_b)):
            ox = idx * (max_w + 4)
            canvas_b[:c.shape[0], ox:ox + c.shape[1]] = c
            cv2.putText(canvas_b, lbl, (ox + 2, max_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        panel_b = canvas_b
    else:
        panel_b = np.zeros((60, 300, 3), dtype=np.uint8)
        cv2.putText(panel_b, "No finger detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ── Panel C: feature bars ──────────────────────────────────────
    if middle is not None:
        nail_feats = calculate_features(nail_crop, mask=mask_kmeans)
        skin_feats = calculate_features(skin_crop, mask=None)
        raw = {f"NAIL_{k}": v for k, v in nail_feats.items()}
        raw.update({f"SKIN_{k}": v for k, v in skin_feats.items()})
        norm = normalize_by_white(raw, wm)

        panel_c = np.zeros((220, 500, 3), dtype=np.uint8)
        draw_percentile_bars(panel_c, norm, y_start=8, x_start=4, w=480, h=200,
                             title="42 Persentil RGB (ternormalisasi)")
    else:
        panel_c = np.zeros((60, 300, 3), dtype=np.uint8)
        cv2.putText(panel_c, "No features", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ── Compose A + B + C ──────────────────────────────────────────
    # align heights: A full-width; B+C side by side below
    bh, bw = panel_b.shape[:2]
    ch_, cw = panel_c.shape[:2]
    bottom_w = bw + 4 + cw
    bottom_h = max(bh, ch_)
    bottom = np.zeros((bottom_h, max(bottom_w, w), 3), dtype=np.uint8)
    bottom[:bh, :bw] = panel_b
    bottom[:ch_, bw + 4: bw + 4 + cw] = panel_c

    # resize panel A width to match bottom
    if w != bottom.shape[1]:
        panel_a = cv2.resize(panel_a, (bottom.shape[1], panel_a.shape[0]), interpolation=cv2.INTER_LINEAR)

    composite = np.vstack([panel_a, bottom])

    cv2.imwrite(str(out_path), composite)
    print(f"  saved: {out_path.name}  ({composite.shape[1]}×{composite.shape[0]})")


def build_composite_mediapipe(pid, img_rgb, landmarks, ordered, middle_idx, out_path):
    """Visualisasi jalur MediaPipe: landmark + 3 box kuku/kulit + mask jari tengah."""
    canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]

    if landmarks is not None:
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)

    middle_mask = None
    if ordered:
        for i, b in enumerate(ordered):
            t, l, bo, r = b.nail_box
            color = NAIL_COLORS[i % 3]
            cv2.rectangle(canvas, (l, t), (r, bo), color, 2)
            cv2.putText(canvas, f"NAIL_{i+1}", (l, max(t - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            t, l, bo, r = b.skin_box
            cv2.rectangle(canvas, (l, t), (r, bo), SKIN_COLORS[i % 3], 1)
        if 0 <= middle_idx < len(ordered):
            nail = ordered[middle_idx].nail_box
            t, l, bo, r = nail
            crop = crop_from_box(img_rgb, ordered[middle_idx].nail_box)
            middle_mask = best_nail_mask(crop, method="auto")
            reg = canvas[t:bo, l:r]
            if middle_mask is not None and reg.size:
                m = cv2.resize(middle_mask, (r - l, bo - t), interpolation=cv2.INTER_NEAREST)
                reg[m > 0] = (255, 120, 120)
                canvas[t:bo, l:r] = cv2.addWeighted(reg, 0.55, canvas[t:bo, l:r], 0.45, 0)

    cv2.putText(canvas, f"MediaPipe | pid={pid} | jari tengah=NAIL_{middle_idx + 1 if middle_idx >= 0 else '-'}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"MediaPipe | pid={pid} | jari tengah=NAIL_{middle_idx + 1 if middle_idx >= 0 else '-'}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
    print(f"  saved: {out_path.name}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gt", "mediapipe"], default="gt")
    args = parser.parse_args()

    if args.mode == "mediapipe":
        from core.hand_landmarks import HandGeometry, HandLandmarkExtractor, derive_all_boxes

        full_dir = cfg.PROJECT_ROOT / "data" / "full_hand"
        if not full_dir.exists():
            print(f"[viz] Folder belum ada: {full_dir}")
            return
        imgs = sorted(p for p in full_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not imgs:
            print(f"[viz] Tidak ada foto di {full_dir}")
            return
        extractor = HandLandmarkExtractor()
        for p in imgs:
            img = load_rgb(str(p))
            landmarks, _score = extractor.detect_ranked(img)
            ordered, middle_idx = [], -1
            if landmarks is not None:
                h, w = img.shape[:2]
                derived = sorted(derive_all_boxes(landmarks, w, h, HandGeometry()),
                                 key=lambda b: (b.nail_box[0] + b.nail_box[2]) / 2, reverse=True)
                ordered = derived
                middle_idx = len(derived) // 2 if derived else -1
            build_composite_mediapipe(p.stem, img, landmarks, ordered, middle_idx,
                                      VIZ_DIR / f"mp_viz_{p.stem}.jpg")
        print(f"\nSemua visualisasi MediaPipe tersimpan di: {VIZ_DIR.resolve()}")
        return

    meta = pd.read_csv(cfg.PROJECT_ROOT / "data" / "metadata.csv")
    det = GTDetector(meta)
    sample_pids = [14, 1, 105]

    for pid in sample_pids:
        row = meta[meta.PATIENT_ID == pid].iloc[0]
        img_path = cfg.PROJECT_ROOT / "data" / "photo" / f"{pid}.jpg"
        if not img_path.exists():
            print(f"  SKIP {pid}: not found")
            continue
        img = load_rgb(str(img_path))
        detected = det.detect_for_patient(pid)
        out = VIZ_DIR / f"viz_{pid}.jpg"
        build_composite(pid, row, img, detected, out)

    print(f"\nSemua visualisasi tersimpan di: {VIZ_DIR.resolve()}")


if __name__ == "__main__":
    main()