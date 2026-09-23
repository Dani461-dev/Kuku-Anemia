# Anemia Kuku — Estimasi Hb dari Foto Kuku

Pipeline AI untuk **estimasi kadar hemoglobin (Hb) dari foto kuku** (single-nail,
tanpa model mata). Mengikuti protokol notebook sumber GitHub, dengan tambahan
pengambilan fitur foto → CSV yang dibangun sendiri.

---

## Asal Data & Lisensi

| | |
|---|---|
| **Sumber dataset** | [biophotonics-msu/photo-haemoglobin](https://github.com/biophotonics-msu/photo-haemoglobin) |
| **Judul dataset** | *Dataset of human skin and fingernails images for non-invasive haemoglobin level assessment* |
| **License** | MIT |
| **Isi dari GitHub** | `data/metadata.csv` + `data/photo/*.jpg` (foto kuku, box kuku/kulit, kadar Hb lab g/L) + notebook `Usage Notes.ipynb` |
| **Tidak disediakan GitHub** | Kode **foto → CSV fitur**. Itu dibangun sendiri di repo ini (`core/build_dataset.py`), mengikuti persis protokol notebook |

> Notebook asli menghitung fitur *in-memory* (tidak disimpan ke CSV). Karena itu
> `core/build_dataset.py` adalah bagian utama repo ini: ia yang menjembatani
> **foto + metadata.csv → CSV fitur → model → prediksi**.

---

## Alur Pipeline

```text
data/photo/*.jpg  +  data/metadata.csv (Hb lab g/L, tanpa box — region dari YOLO-seg)
      │
      ▼
core/build_dataset.py          │ 1. crop ingin (box jari tengah)
      │                        │ 2. white reference: median img[350:400,300:350]
      ▼                        │ 3. 42 fitur persentil RGB (NAIL_ + SKIN_):
core/outputs/features_*.csv    │    P5,15,25,50,75,85,95 × R,G,B × 2 region
      │                        │ 4. normalisasi: fitur / white median per channel
      ▼
core/train_hb.py               │ RobustScaler + ElasticNet
      │                        │ --protocol nested (KANONIK) = improved (MAE)
      ▼                        │ --protocol notebook = reproduksi notebook asli
core/models/                   │ elasticnet_model.joblib + model_metadata.json (kanonik)
      │                        │ + notebook_baseline/ (reproduksi) + improved_all250/
      ▼
core/inference.py              │ predict_hb() → estimasi Hb g/dL + kategori WHO
(pakai model terlatih)         │ threshold: wanita <12.0, pria <13.0 g/dL
```

**Cara biasa memakai box**: dataset menyediakan `NAIL_BOUNDING_BOXES` /
`SKIN_BOUNDING_BOXES` per pasien (jari tengah = `NAIL_2`/`SKIN_2`) → inilah
"ground truth" yang dipakai untuk latih model.

---

## Peta Folder

```text
anemia-app/
├── core/                   # PAKET INTI (pipeline lengkap)
│   ├── build_dataset.py    #   foto + metadata → CSV fitur (kontribusi utama)
│   ├── features.py         #   42 fitur persentil + normalisasi white
│   ├── extended.py         #   fitur HSV/LAB + kontras (eksperimen, TIDAK dipakai)
│   ├── masking.py          #   mask kuku (kmeans/otsu/grabcut)
│   ├── detectors.py        #   GT (dataset) / YOLO (eksperimen) / Light
│   ├── hand_landmarks.py   #   MediaPipe HandLandmarker → box (runtime full-hand)
│   ├── train_hb.py         #   latih model (nested / notebook protocol)
│   ├── train_fullhand.py   #   retrain model pada dataset full-hand (T3)
│   ├── inference.py        #   prediksi Hb + kategori WHO
│   ├── validate_mediapipe.py  # harness validasi jalur runtime (T1/T2)
│   ├── visualize.py        #   visualisasi box/mask/fitur (mode gt & mediapipe)
│   ├── categorize.py       #   helper ambang WHO
│   ├── calibration.py      #   deteksi kartu warna ArUco (white reference)
│   ├── chart_generator.py  #   generate kartu warna
│   ├── config.py, util.py, __init__.py
│   ├── models/             #   MODEL TERLATIH (kanonik + baseline) ⭐
│   ├── outputs/            #   CSV fitur hasil build_dataset
│   ├── assets/             #   model MediaPipe (.task)
│   └── chart_output/       #   kartu warna keluaran
├── data/
│   ├── metadata.csv        #   LEAN: PATIENT_ID + MEASUREMENT_DATE + Hb lab (tanpa box)
│   ├── legacy_metadata_with_boxes.csv  #   arsip metadata ber-box (untuk evaluasi GT) 
│   ├── photo/              #   250 foto kuku (1 foto ↔ 1 baris metadata)
│   ├── photo_unlabelled/   #   foto tanpa Hb lab (mis. 185.jpg), tidak dipakai
│   ├── full_hand/          #   foto tangan penuh user (untuk uji runtime & T3)
│   ├── full_hand_template.csv  # template metadata dataset full-hand (T3)
│   └── output/             #   hasil validasi (predictions.csv)
├── docs/
│   ├── PROTOKOL_NOTEBOOK.md           # perbandingan core/ vs notebook asli
│   ├── PROTOKOL_FULLHAND.md           # protokol pengumpulan dataset full-hand (T3)
│   ├── NAIL_ANALYSIS_PIPELINE.md      # dok pipeline (awal)
│   ├── ANEVIA_FUSION_CALCULATION.md   # referensi fusion mata+kuku (di luar scope)
│   └── Usage Notes.ipynb              # salinan notebook asli (referensi)
├── experiments/yolo/       # detektor YOLO OPSIONAL (belum dilatih)
│   ├── prepare_yolo.py     #   metadata → dataset YOLO
│   ├── verify_annots.py    #   visual cek anotasi
│   ├── train.py            #   training YOLO11n
│   ├── dataset/            #   data.yaml + images/ + labels/
│   └── verification/       #   preview anotasi (250 gambar)
├── models/
│   └── legacy_74features/  # arsip skema 74 fitur (percobaan lama, TIDAK dipakai)
├── weights/
│   └── yolo11n.pt          # base weight untuk training YOLO (opsional)
└── tests/                  # sanity test
```

---

## Perintah Cepat

```bash
# 1) Ekstraksi fitur (default: GT boxes, white=fixed, mask=kmeans/otsu)
python3 core/build_dataset.py --boxes gt --white fixed            # masked
python3 core/build_dataset.py --boxes gt --white fixed --no-mask # = protokol notebook

# 2) Latih model
#    model KANONIK = improved (masked, semua 250 data, nested CV) — unggul MAE
python3 core/train_hb.py --features core/outputs/features_gt_fixed.csv --no-balance
#    reproduksi baseline asli GitHub (notebook, nomask, balanced-100, GridSearchCV)
python3 core/train_hb.py --protocol notebook \
    --features core/outputs/features_gt_fixed_nomask.csv \
    --out-dir core/models/notebook_baseline

# 3) Prediksi (smoke test) — jalur GT legacy (box dari arsip; kanonik runtime = seg26)
python3 -c "
import pandas as pd
from core.inference import predict_hb
from core.detectors import GTDetector
from core.pipeline import select_middle_finger
meta = pd.read_csv('data/legacy_metadata_with_boxes.csv')
det = GTDetector(meta)
pid = 14
mid = select_middle_finger(det.detect_for_patient(pid))
print(predict_hb(f'data/photo/{pid}.jpg', mid.nail_box, mid.skin_box, gender='female'))
"
```

---

## Alur Runtime (foto tangan penuh → Hb) — MediaPipe

Pada aplikasi nyata, **tidak ada box di metadata**. Jalur runtime:

```
foto tangan penuh (background bebas)
   → MediaPipe HandLandmarker → 3 jari (NAIL_1/2/3, SKIN_1/2/3)  ← format = metadata
   → pilih jari tengah (NAIL_2)
   → mask kuku (kmeans/otsu) 
   → white reference: KARTU WARNA ArUco (dari core/chart_generator.py) → fallback auto
   → 42 fitur persentil ternormalisasi
   → model → estimasi Hb g/dL → kategori WHO
```

**Jalur runtime alternatif — YOLO26-seg (tracking + masking per-pixel):**

1. Latih model: `python3 experiments/yolo26_seg/train_seg26.py --epochs 60`
2. Jalankan harness: `python3 core/run_seg_pipeline.py --input-dir data/full_hand --device 0`
3. Overlay + ringkasan CSV → `core/outputs/viz_seg/` & `data/output/seg_predictions.csv`

```
foto → YOLO26-seg (mask per-pixel per kuku)
   → box NAIL (dari mask) + box SKIN (geometri: geser +2.3× lebar, konsisten layout MSU)
   → pilih jari tengah (median vertikal)
   → mask kuku → 42 fitur persentil ternormalisasi (white=auto)
   → model selaras (core/models/seg_runtime) → estimasi Hb g/dL → kategori WHO
```

Model Hb **selaras runtime** (`core/models/seg_runtime`, dibangun `core/build_features_seg.py`)
dipakai otomatis oleh pipeline seg26 — fitur training ≡ fitur runtime.
Deteksi bisa dipilih: `make_detector("seg26")` (mask+box), `"yolo"`, `"mediapipe"`, `"light"`.
Bobot seg26 override: env `ANEVIA_SEG26_WEIGHTS` (default: `experiments/yolo26_seg/runs/seg26/weights/best.pt`).
Override model Hb: env `ANEVIA_HB_MODEL_DIR`.

**Guard kewajaran** (`core/guard.py`): tiap prediksi diperiksa — ada kuku,
mask coverage box 5–99%, ≥70% fitur dalam rentang training, Hb ∈ 4–18 g/dL.
Gagal guard → `FLAGS_OK=0` + alasan di `ISSUES`, Hb tidak dikeluarkan (arahkan
"ulangi foto"). White-source dikunci dari metadata model (default `auto`); jika CLI
menyimpang akan ada warning.

**Validasi jalur runtime (T1/T2):**

```bash
# 1) Siapkan foto tangan penuh di data/full_hand/ (idealnya kartu warna dalam frame)
# 2) Jalankan harness:
python3 core/validate_mediapipe.py --gender female
#    → core/outputs/viz/mediapipe_*.jpg  (overlay landmark+box+mask)
#    → data/output/predictions.csv       (format metadata + prediksi + flag kewajaran)
# 3) Lihat overlay visual per-foto:
python3 core/visualize.py --mode mediapipe
```

**Flag kewajaran (T2)**: Hb ∈ 4–18 g/dL · mask coverage 15–99% · confidence hand ≥ 0.5 ·
fitur dalam rentang training ≥ 80%. `FLAGS_OK=1` = semua lolos.

**Konfigurasi deteksi runtime (seg26, dua-pass)**: pass utama `conf=0.30` @ `imgsz 640`
(skala training — di foto resolusi tinggi memberi 5 kuku bersih conf 0.34–0.79). Jika
< 3 kuku (foto kecil/compressed/framing global): fallback `conf=0.10` @ `imgsz 1280`
+ dedupe duplikat (IoU>0.5) + min-conf 0.25 + cap 5. Tanpa dua-pass, conf rendah
menghasilkan duplikat palsu (kuku 5 terlihat 6–9). CLI: `run_seg_pipeline.py`
(default dua-pass aktif; `--no-fallback` untuk nonaktifkan).

**Retrain pada dataset full-hand (T3)** — lihat `docs/PROTOKOL_FULLHAND.md`:
- Metadata lean (`data/full_hand_template.csv`), **tanpa box** — region dari YOLO26-seg.
- `core/train_fullhand.py` membandingkan 2 strategi & menyimpan terbaik:
  A) full-hand saja · B) MSU + full-hand (gabung).

```bash
python3 core/train_fullhand.py --conf 0.15 --device 0
```

**Eksperimen model lain (selain ElasticNet)** — `experiments/hb_models/benchmark_seg.py`
pada fitur seg-runtime: coba RF/ET/GBR/HistGB/SVR/KNN/MLP/XGB/LightGBM/Stacking.
Hasil: **ElasticNet (15.91 g/L) tetap terbaik**; semua non-linear/jar-jauh ≥16.2 g/L.
→ baseline dipertahankan (data, bukan model, yang menjadi pembatas akurasi).

---

## Model (ringkas performa)

| Model | Protokol | Metrik |
|---|---|---|
| **`core/models/seg_runtime/` (RE-BASELINE)** | **selaras runtime** — fitur YOLO26-seg mask + skin geometri + white auto (250 pasien, nested CV) | **MAE 15.96 g/L (1.60 g/dL)** · RMSE 20.67 · R² 0.40 — dipakai otomatis oleh pipeline seg26 |
| **`core/models/elasticnet_model.joblib` (KANONIK)** | **improved** (masked, semua 250 pasien, nested CV) | **MAE 15.99 g/L (1.60 g/dL)** · RMSE 20.62 g/L · R² 0.40 |
| `core/models/notebook_baseline/` | reproduksi notebook asli (nomask, balanced-100, GridSearchCV) | alpha 0.2057, l1 0.9 · test RMSE 20.26 g/L · bias test −4.3, LoA (−43, +34) g/L |
| `core/models/improved_all250/` | duplikat kanonik (cadangan) | MAE 1.60 g/dL · RMSE 2.06 g/dL · R² 0.40 |

Estimasi dalam **g/L** (dikonversi g/dL di API). Kategori WHO: wanita <12.0 g/dL,
pria <13.0 g/dL.

---

## Catatan Penting / Keterbatasan

1. **MediaPipe tidak mendeteksi tangan di foto dataset** (tangan 3 jari masuk dari
   tepi kanan, tanpa telapak lengkap → gagal). Kanonik runtime kini **YOLO26-seg**
   (deteksi+mask, dilatih pada foto full-hand V2). Box GT di
   `data/legacy_metadata_with_boxes.csv` hanya untuk evaluasi/reproduksi.
2. **YOLO belum dilatih** — hanya ada sanity run 1 epoch. Lihat
   `experiments/yolo/train.py`.
3. **Fitur HSV/LAB + kontras (74 fitur) MERUGIKAN akurasi** di uji empiris
   (CV RMSE 38.9 vs 23.4) → diarsipkan di `models/legacy_74features/`.
4. **Batas akurasi = jumlah data, bukan pilihan model.** Eksperimen benchmark
   (`experiments/hb_models/benchmark.py`, hasil di `core/outputs/experiment_results.csv`):
   model non-linear (RF/GBR/SVR/Huber), tuning grid-lebar, ensembel, target-transform,
   bagging KDE, dan augmentasi fitur SEMUANYA tidak melampaui ElasticNet secara berarti
   (semua beda ≤ ~0.03 g/dL → tingkat noise). Jalan naik akurasi = dataset lebih banyak /
   full-hand (T3).
5. **Hasil adalah estimasi AI**, bukan pengukuran lab; interpretasi medis wajib
   ditegaskan lewat tes klinis.

Detail verifikasi kesesuaian dengan notebook asli: lihat
[`docs/PROTOKOL_NOTEBOOK.md`](docs/PROTOKOL_NOTEBOOK.md).