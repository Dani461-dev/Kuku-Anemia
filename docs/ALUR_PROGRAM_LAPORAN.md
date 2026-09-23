# Alur Kerja Sistem Estimasi Hb dari Foto Kuku

> Ringkasan untuk laporan: bagaimana program bekerja dari foto masuk sampai
> estimasi kadar hemoglobin (Hb) keluar.

## A. Gambaran Umum

Sistem menerima **foto kuku/tangan penuh** sebagai input, lalu mengeluarkan
**estimasi kadar hemoglobin (Hb) dalam g/dL** beserta **kategori anemia WHO**
(Normal / Rendah-Anemia). Threshold WHO dewasa: **wanita < 12.0 g/dL**,
**pria < 13.0 g/dL**.

Sistem dibagi menjadi **2 fase**:

1. **Fase Pelatihan (Offline)** — membangun model regresi Hb dari dataset ber-label.
2. **Fase Inferensi (Runtime)** — memakai model terlatih untuk foto baru tanpa label.

---

## B. Fase Pelatihan (Offline)

```
data/photo/*.jpg  +  data/metadata.csv (Hb lab g/L)
        │
        ▼   (1) Deteksi & segmentasi
YOLO26-seg → box NAIL per kuku + mask per-pixel
        │      box SKIN = geometri (geser +2.3× lebar kuku)
        ▼   (2) Pilih jari tengah (median vertikal box NAIL)
        ▼   (3) Ekstraksi 42 fitur persentil RGB
build_features_seg.py → core/outputs/features_seg26_auto.csv
        │     (NAIL + SKIN, persentil P5,15,25,50,75,85,95 × R,G,B)
        │     (normalisasi: nilai ÷ median white-reference per channel)
        ▼   (4) Training model
train_hb.py
    - RobustScaler (standarisasi robust terhadap outlier)
    - ElasticNetCV (regresi linier ber-regularisasi L1+L2)
    - Evaluasi: Nested 5-fold CV (tuning di dalam tiap fold)
        ▶ hasil: core/models/seg_runtime/elasticnet_model.joblib
                 + model_metadata.json (urutan fitur, threshold WHO, metrik)
```

Titik penting: `metadata.csv` hanya dipakai sebagai **target Hb lab (g/L)**;
region (box) sepenuhnya berasal dari deteksi YOLO26-seg → fitur pelatihan
**identik/selaras** dengan fitur runtime.

---

## C. Fase Inferensi (Runtime)

```
Foto tangan penuh baru
        │
        ▼   (1) Deteksi + segmentasi (YOLO26-seg dua-pass: conf 0.30@640 → fallback 0.10@1280 + dedupe)
        │      └ mask per-pixel per kuku + box NAIL + box SKIN geometri
        ▼   (2) Pilih jari tengah = instance dengan pusat vertikal box NAIL
        │      yang merupakan median
        ▼   (3) Mask kuku (dari seg) + region kulit (box geometri dikurangi
        │      mask kuku)
        ▼   (4) White reference (koreksi pencahayaan)
        │      - "auto"  : deteksi piksel terang (HSV) → median RGB
        │      - "chart" : kartu warna ArUco → patch putih (fallback auto)
        ▼   (5) 42 fitur persentil RGB ternormalisasi
        │      (NAIL_ + SKIN_ × R,G,B × 7 persentil) ÷ white median/channel
        ▼   (6) Prediksi model (ElasticNet): Hb g/L → ÷10 → g/dL
        ▼   (7) Klasifikasi WHO + Guard kewajaran (plausibility check)
        ▼
Hasil: Hb g/dL + kategori + flag kewajaran (FLAGS_OK)
```

---

## D. Guard Kewajaran (Filter "Prediksi Liar")

Sebelum hasil dikeluarkan, `core/guard.py` memeriksa:

- **Ada kuku terdeteksi** (jumlah instance > 0)
- **Coverage mask** kuku terhadap box dalam 5–99%
- **≥70% dari 42 fitur** berada dalam rentang persentil 1–99 data pelatihan
  (anti drift fitur)
- **Hb prediksi dalam rentang fisiologis 4–18 g/dL**

Gagal satu saja → `FLAGS_OK = 0` + alasan di `ISSUES`, Hb **tidak
dikeluarkan** (dianjurkan "ulangi foto").

---

## E. Fitur yang Dipakai (42 Fitur)

7 persentil intensitas (P5, 15, 25, 50, 75, 85, 95) × 3 channel (R, G, B) ×
2 region (NAIL dan SKIN) = 42 fitur. Persentil dihitung **hanya atas piksel
mask** (kuku), menggantikan crop 60% tengah dari notebook asli.

---

## F. Model & Hasil

- **Model**: RobustScaler + ElasticNet (regresi; alpha & l1_ratio dipilih CV)
- **Metrik**: MAE **15.96 g/L (1.60 g/dL)**, RMSE 20.67 g/L, R² 0.40
  (nested CV, 250 pasien)
- Eksperimen model lain (RF/SVR/XGB/MLP, dll.) **tidak melampaui** ElasticNet
  → baseline dipertahankan.

---

## G. Batasan

- Hasil adalah **estimasi AI**, bukan pengukuran lab; interpretasi medis wajib
  dikonfirmasi lewat tes klinis.
- Model Hb dilatih pada foto MSU 3-jari terpotong; akurasi pada foto full-hand
  aplikasi belum terukur (butuh data: foto full-hand + Hb lab).