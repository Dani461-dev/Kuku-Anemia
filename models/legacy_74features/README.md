# Legacy — Percobaan Ekstraksi Fitur 74 Dimensi (Arsip)

Folder ini berisi **percobaan pertama** ekstraksi fitur yang dibuat sendiri
(sebelum memakai protokol notebook asli dari dataset GitHub).

- `extract_features.py` — skema 74 fitur HSV/LAB/chromaticity + kontras.
- `elasticnet_model.joblib`, `robust_scaler.joblib`, `model_metadata.json`,
  `feature_coefficients.csv`, `X_raw.npy`, `y_raw.npy` — artefak lama.

**Mengapa diarsipkan:**
- Model hasil skema ini "mati" (`n_nonzero_coef = 0` — semua koefisien nol).
- Skema fiturnya **BUKAN** dari `Usage Notes.ipynb` sumber GitHub
  (`biophotonics-msu/photo-haemoglobin`), melainkan desain tim sendiri.
- Setelah diuji empiris, **memperburuk akurasi** dibanding protokol notebook
  (42 fitur persentil RGB ternormalisasi white-reference).

**Pipeline kanonik ada di `core/`** (protokol = notebook asli):
- Ekstraksi fitur foto → CSV: `core/build_dataset.py`
- Pelatihan: `core/train_hb.py` (RobustScaler + ElasticNet)
- Prediksi: `core/inference.py`
- Model terlatih: `core/models/`

Hanya referensi. Jangan dihubungkan ke aplikasi.