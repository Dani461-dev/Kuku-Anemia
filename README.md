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
data/photo/*.jpg  +  data/metadata.csv (NAIL_2/SKIN_2 boxes, Hb lab g/L)
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
│   ├── metadata.csv        #   dari GitHub (Hb lab g/L + box)
│   ├── photo/              #   250 foto kuku dari GitHub
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

# 3) Prediksi (smoke test)
python3 -c "
import pandas as pd
from core.inference import predict_hb
from core.detectors import GTDetector
from core.pipeline import select_middle_finger
meta = pd.read_csv('data/metadata.csv')
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

**Retrain pada dataset full-hand (T3)** — lihat `docs/PROTOKOL_FULLHAND.md`:

```bash
python3 core/train_fullhand.py
```

---

## Model (ringkas performa)

| Model | Protokol | Metrik |
|---|---|---|
| **`core/models/elasticnet_model.joblib` (KANONIK)** | **improved** (masked, semua 250 pasien, nested CV) | **MAE 15.99 g/L (1.60 g/dL)** · RMSE 20.62 g/L · R² 0.40 |
| `core/models/notebook_baseline/` | reproduksi notebook asli (nomask, balanced-100, GridSearchCV) | alpha 0.2057, l1 0.9 · test RMSE 20.26 g/L · bias test −4.3, LoA (−43, +34) g/L |
| `core/models/improved_all250/` | duplikat kanonik (cadangan) | MAE 1.60 g/dL · RMSE 2.06 g/dL · R² 0.40 |

Estimasi dalam **g/L** (dikonversi g/dL di API). Kategori WHO: wanita <12.0 g/dL,
pria <13.0 g/dL.

---

## Catatan Penting / Keterbatasan

1. **MediaPipe tidak mendeteksi tangan di foto dataset** (tangan 3 jari masuk dari
   tepi kanan, tanpa telapak lengkap → gagal). Karena itu box `metadata.csv`
   (ground truth dataset) dipakai untuk latih. MediaPipe tetap berguna untuk foto
   asli pengguna (tangan penuh).
2. **YOLO belum dilatih** — hanya ada sanity run 1 epoch. Lihat
   `experiments/yolo/train.py`.
3. **Fitur HSV/LAB + kontras (74 fitur) MERUGIKAN akurasi** di uji empiris
   (CV RMSE 38.9 vs 23.4) → diarsipkan di `models/legacy_74features/`.
4. **Hasil adalah estimasi AI**, bukan pengukuran lab; interpretasi medis wajib
   ditegaskan lewat tes klinis.

Detail verifikasi kesesuaian dengan notebook asli: lihat
[`docs/PROTOKOL_NOTEBOOK.md`](docs/PROTOKOL_NOTEBOOK.md).