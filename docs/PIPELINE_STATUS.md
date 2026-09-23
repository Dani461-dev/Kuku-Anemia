# Status Pipeline — Anevia (Nail Tracking + Masking + Hb)

> Status akurat per 18 Sep 2026. Dokumen ini merekam keputusan, artefak, serta
> gap yang tersisa, sebagai referensi tunggal sebelum lanjut pengembangan.

## Arsitektur final

```
Foto (tangan penuh, background bebas)
  → YOLO26-seg            [tracking + masking]
      ├─ box NAIL per kuku (dari mask)
      ├─ mask kuku per-pixel
      └─ box SKIN (geometri: geser +2.3× lebar — konsisten layout training)
  → pilih jari tengah (median vertikal dari instance seg)
  → 42 fitur persentil RGB (white=auto)
  → Model Hb selaras (ElasticNet re-baseline `seg_runtime`)
  → Hb g/dL → kategori WHO (wanita <12.0, pria <13.0 g/dL)
```

## Status per tahap

### 1. Tracking + Masking — ✅ Kuat & sudah pada domain aplikasi

Data training **NailSegmentationDatasetV2** (`Data Tambahan/archive (12)`):
didominasi **full-hand** (train 93%, val 84%, test 91%; sisanya macro kuku).
Artinya model **sudah dilatih pada foto tangan penuh** yang setara runtime aplikasi.

| Metrik | Nilai | Catatan |
|---|---|---|
| Mask mAP50 (val) | **0.965** | |
| Mask mAP50-95 (val) | 0.711 | |
| V2 test | Dice 0.847 · IoU 0.756 · deteksi 98.3% | on-domain |
| Cross-domain MSU | deteksi 100% · GT ≥IoU.5 = 98.9% · IoU mean 0.750 | conf 0.15 |
| Kecepatan | ~10 ms/foto (RTX 2050) | |

### 2. Ekstraksi fitur — ✅ Selaras training ≡ runtime

- Satu jalur ekstraksi: **seg mask + skin geometri + white auto**
  (`core/build_features_seg.py`).
- `metadata.csv` adalah **metadata lean** (PATIENT_ID + MEASUREMENT_DATE + Hb lab)
  dan hanya dipakai sebagai **sumber target Hb lab** — region sepenuhnya dari YOLO-seg.
- Kontrak **1 foto ↔ 1 pasien**: `data/photo/{PATIENT_ID}.jpg` = 250 foto ↔ 250 baris
  (validasi otomatis di `build_features_seg`). Foto tanpa lab diarsipkan ke
  `data/photo_unlabelled/`.
- Metadata ber-box diarsipkan ke `data/legacy_metadata_with_boxes.csv`
  (dipakai `eval_seg_msu.py`, `GTDetector`, `build_dataset --boxes gt` — bukan kanonik).
- Produk fitur: `core/outputs/features_seg26_seg26_auto.csv`
  (250 subjek · 42 fitur · mask coverage box mean 0.89 · tanpa NaN).

### 3. Model Hb (re-baseline `seg_runtime`) — ⚠️ Berfungsi, belum validasi full-hand

| Aspek | Nilai |
|---|---|
| Nested CV MAE | **15.96 g/L (1.60 g/dL)** |
| CV RMSE / R² | 20.67 g/L / 0.40 |
| Pearson | 0.633 |
| Fitur training ≡ runtime | ✅ (re-baseline) |
| Model aktif | `core/models/seg_runtime/` — dipilih otomatis `inference._model_paths()` |
| **Gap** | Model dilatih dari foto **MSU 3-jari terpotong** (ber-label lab). Foto **full-hand belum ada nilai Hb lab** → akurasi Hb pada foto aplikasi asli **belum terukur** (T3). |

## Artefak

| Artefak | Path |
|---|---|
| Bobot YOLO26-seg | `experiments/yolo26_seg/runs/seg26/weights/best.pt` |
| Dataset YOLO (6.844 img · 29.641 instance) | `experiments/yolo26_seg/dataset/` + `~/yolo_dataset/` (lokal) |
| Script dataset / training | `experiments/yolo26_seg/prepare_seg26.py` · `train_seg26.py` |
| Evaluasi | `core/eval_seg_msu.py` · `core/eval_seg_v2.py` |
| Build fitur runtime | `core/build_features_seg.py` |
| Train model Hb | `core/train_hb.py` → `core/models/seg_runtime/` |
| Detektor + inferensi + harness | `core/seg_detector.py` · `core/inference.py` · `core/run_seg_pipeline.py` |

## Gap & prioritas lanjut

1. ✅ **Model lain diuji** — `experiments/hb_models/benchmark_seg.py` (fitur seg):
   ElasticNet/Ridge 15.91 g/L (terbaik), SVR 16.25, RF 16.70, XGB/HistGB ~17.3,
   MLP 32.6, Stacking 15.93 → **baseline ElasticNet dipertahankan**.
2. ✅ **Paket T3 siap** — template lean (`data/full_hand_template.csv` tanpa box),
   `core/train_fullhand.py` jalur YOLO-seg + strategi gabung (MSU+full-hand),
   SOP di `docs/PROTOKOL_FULLHAND.md`.
3. ⏳ **Data T3 = kunci berikutnya**: foto full-hand + Hb lab (min 50, ideal 100+)
   → jalankan `train_fullhand.py` → bandingkan CV MAE vs 15.96 g/L.
4. (Opsional) Ensemble prediksi 3 jari — sudah diuji, TIDAK membantu (14.93 vs
   16.96 g/L rata2-jari).

## Fine-tune seg26 ke domain foto aplikasi (proses aktif)

- **Masalah**: seg26 dilatih hanya dgn `archive (12)/NailSegmentationDatasetV2`
  (kuku dekat); di foto app (tangan penuh + bg bebas) confidence turun
  (0.30–0.75 vs 0.7–0.9 di domain asli) walau deteksi 5/5/5.
- **F1 — pseudo-label**: `experiments/yolo26_seg/prepare_domain_labels.py`
  (jalankan per batch foto baru di `data/full_hand/`):
  seg26 dua-pass → mask conf ≥0.30 → QA otomatis (≥4 kuku, box ≥12px,
  fill 30–95%, non-overlap) → poligon YOLO-seg di `dataset_app/`
  + `data_app.yaml` (V2 + app).
- **F2 — fine-tune**: `train_seg26.py --model runs/seg26/weights/best.pt
  --data data_app.yaml --epochs 30 --imgsz 640 --device 0 --scale 0.9`.
- **F3 — gate**: `core/eval_gate_seg26.py --new runs/seg26{2}/weights/best.pt`
  (V2 Dice ≥ 0.80 & tidak turun >0.02 · app 5/5 & mean conf naik · MSU tidak
  regresi). Lolos → swap weights runtime; gagal → pertahankan dua-pass.

## Keterbatasan / disclaimer

- Hasil Hb adalah **estimasi AI**, bukan pengukuran lab; interpretasi medis wajib
  dikonfirmasi lewat tes klinis.
- Konfigurasi runtime aman: `Yolo26SegDetector(conf=0.30, imgsz=640, fallback=True
  -> conf 0.10 @ imgsz 1280 + dedupe utk foto susah)`, `white=auto`,
  model `seg_runtime`. Keyakinan tinggi pada tracking+masking (full-hand trained);
  keyakinan sedang pada nilai Hb (belum ada ground-truth full-hand).