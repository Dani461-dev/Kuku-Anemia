# PROTOKOL NOTEBOOK — Kesesuaian `core/` dengan Notebook Asli

Dokumen ini memverifikasi bahwa pipeline `core/` adalah **port setia** dari
`Usage Notes.ipynb` milik [biophotonics-msu/photo-haemoglobin](https://github.com/biophotonics-msu/photo-haemoglobin),
dan menjelaskan varian "improved" yang dipakai sebagai model kanonik.

---

## 1. Tabel Paritas (core/ vs Notebook)

| Parameter | Notebook asli | `core/` | Status |
|---|---|---|---|
| Region white reference | `img[350:400, 300:350]`, median per channel | `core/config.py` `WHITE_ROWS=(350,400)`, `WHITE_COLS=(300,350)`; `pipeline.white_fixed` | ✓ sama |
| `cut_image` | low=0.2, high=0.8 (60% tengah) | `core/features.py::cut_image` | ✓ sama |
| Percentile | `[5, 15, 25, 50, 75, 85, 95]` | `core/config.py` `PERCENTILE_LEVELS` | ✓ sama |
| Channel | R, G, B | `core/config.py` `COLORS="RGB"` | ✓ sama |
| Jari yang dipakai | hanya jari tengah (`NAIL_2`, `SKIN_2`) | `core/pipeline.py::select_middle_finger` | ✓ sama |
| Nama fitur | `NAIL_R_p=5` … `SKIN_B_p=95` | `core/features.py::feature_names` + prefix | ✓ sama |
| Total fitur | 42 (21 nail + 21 skin) | `core/train_hb.py::load_features` | ✓ sama |
| Target | `HB_LEVEL_GperL` (g/L), tanpa konversi | `core/train_hb.py` | ✓ sama |
| Normalisasi | fitur / white-median per channel | `core/features.py::normalize_by_white` | ✓ sama |
| Model | `Pipeline(RobustScaler, ElasticNet(max_iter=10000))` | `core/train_hb.py` | ✓ sama |
| Balancing | `gaussian_kde(bw=0.5)`, `weights=1/kde`, n=100, seed 42 | `core/train_hb.py::balance` | ✓ sama |
| GridSearch | `l1_ratio [0.01, 0.1, 0.5, 0.9, 0.99]`, `alpha logspace(-4,4,100)` | `core/train_hb.py::notebook_gridsearch` | ✓ sama |
| CV | `KFold(n_splits=7)`, skor RMSE | `core/train_hb.py` | ✓ sama |

> Catatan sklearn-1.9: notebook memakai `mean_squared_error(..., squared=False)`;
> API itu dihapus di sklearn ≥1.4, jadi `core/` memakai `root_mean_squared_error`
> (hasil identik).

---

## 2. Bukti Reproduksi Baseline (notebook)

Reproduksi (diarsipkan sebagai `core/models/notebook_baseline/`):

```bash
python3 core/train_hb.py --protocol notebook \
    --features core/outputs/features_gt_fixed_nomask.csv \
    --out-dir core/models/notebook_baseline
```

Hasil yang terverifikasi (sama angka dengan notebook):

| Metrik | Notebook | `core/` (`--protocol notebook`) |
|---|---|---|
| best alpha | 0.20565123083486536 | 0.20565123083486536 |
| best l1_ratio | 0.9 | 0.9 |
| best CV RMSE (GridSearch) | 23.856 g/L | 23.856 g/L |
| Test RMSE | ~20.26 g/L | 20.262316335342994 g/L |
| Val bias / LoA | 0.4 / (−46.6, 47.5) g/L | 0.4 / (−46.6, 47.5) g/L |
| Test bias / LoA | −4.3 / (−43.1, 34.5) g/L | −4.3 / (−43.1, 34.5) g/L |

> Determinisme juga terverifikasi: fitur `build_dataset --no-mask` vs hitungan
> in-memory notebook sel 17 → **identik** (selisih maks ~2e-16).

---

## 3. Model KANONIK = "Improved" (nested)

Protokol default `core/train_hb.py --protocol nested`:

| Aspek | Notebook (baseline) | Improved (nested, KANONIK) |
|---|---|---|
| Data latih | subset KDE-balanced 100 | semua 250 pasien (`--no-balance`) |
| Tuning | GridSearchCV 7-fold (500 kandidat) | nested CV di dalam outer KFold(5) |
| Estimator akhir | ElasticNet | ElasticNetCV |
| Masking kuku | tanpa mask (crop polos) | mask (kmeans/otsu) — *opsional* |
| Hasil (CV) | RMSE 23.86 g/L | MAE 15.99 g/L, RMSE 20.62 g/L |

Model kanonik: `core/models/elasticnet_model.joblib` (+ `model_metadata.json`).
Baseline notebook diarsipkan ke `core/models/notebook_baseline/`.

---

## 4. Eksperimen lain yang DIARSIPKAN

| Eksperimen | Hasil | Status |
|---|---|---|
| Skema 74 fitur (HSV/LAB/chromaticity + kontras) | model mati (`n_nonzero_coef=0`); lalu diuji: CV RMSE 38.9 g/L (lebih buruk dari 23.4) | `models/legacy_74features/` (referensi) |
| Deteksi MediaPipe di foto dataset | tidak mendeteksi (tangan tak lengkap) | tetap tersedia untuk foto asli |
| Deteksi YOLO | belum dilatih (hanya 1 epoch) | `experiments/yolo/` (opsional) |

---

## 5. Keterbatasan & Catatan Juri

- Hasil estimasi adalah **AI-estimated Hb**, bukan pengukuran lab.
- Evaluasi memakai **RMSE/MAE/R²** pada skala g/L; konversi g/dL = g/L ÷ 10.
- Ambang anemia WHO: wanita <120 g/L, pria <130 g/L.
- Reproduksibilitas: `data/` tidak dimodifikasi; seluruh artefak bisa
  digenerate ulang dari `core/build_dataset.py` → `core/train_hb.py`.