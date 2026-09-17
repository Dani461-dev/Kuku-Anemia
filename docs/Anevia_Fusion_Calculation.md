# Anevia — Fusion Calculation for Eye and Nail Models

## Overview

Pada project **Anevia — AI-Powered Anemia Screening from Eye and Nail Images**, model citra mata dan model citra kuku dilatih secara terpisah.

Masing-masing model menghasilkan estimasi kadar hemoglobin:

```text
Eye Model  → Estimated Hb from eye image
Nail Model → Estimated Hb from nail image
```

Karena kedua model memiliki performa yang berbeda, hasil akhir sebaiknya tidak langsung digabung menggunakan rata-rata 50:50.

Metode yang direkomendasikan adalah:

> **Weighted Late Fusion based on validation performance**

Artinya, model yang memiliki error lebih kecil saat validasi akan mendapat bobot lebih besar.

---

# 1. Input dari Kedua Model

Contoh:

```text
Prediksi Model Mata = 10.8 g/dL
Prediksi Model Kuku = 11.4 g/dL
```

Hasil evaluasi validasi:

```text
MAE Model Mata = 0.70
MAE Model Kuku = 1.00
```

Karena MAE mata lebih kecil, model mata dianggap lebih akurat dan akan mendapatkan bobot lebih besar.

---

# 2. Menghitung Bobot Awal

Gunakan inverse error:

```text
raw_weight_eye  = 1 / MAE_eye
raw_weight_nail = 1 / MAE_nail
```

Dengan contoh:

```text
raw_weight_eye  = 1 / 0.70
                 = 1.4286

raw_weight_nail = 1 / 1.00
                 = 1.0000
```

---

# 3. Normalisasi Bobot

Bobot harus dinormalisasi agar jumlahnya sama dengan 1.

Formula:

```text
weight_eye =
raw_weight_eye /
(raw_weight_eye + raw_weight_nail)
```

```text
weight_nail =
raw_weight_nail /
(raw_weight_eye + raw_weight_nail)
```

Hasil:

```text
weight_eye =
1.4286 / (1.4286 + 1.0000)
≈ 0.588

weight_nail =
1.0000 / (1.4286 + 1.0000)
≈ 0.412
```

Jadi:

```text
Eye Model  = 58.8%
Nail Model = 41.2%
```

---

# 4. Menghitung Estimasi Hb Final

Formula:

```text
Final Hb =
(Eye Hb × Eye Weight)
+
(Nail Hb × Nail Weight)
```

Contoh:

```text
Final Hb =
(10.8 × 0.588)
+
(11.4 × 0.412)
```

Perhitungan:

```text
Final Hb =
6.3504
+
4.6968

Final Hb ≈ 11.05 g/dL
```

Jadi hasil akhir Anevia:

```text
Eye Prediction   : 10.8 g/dL
Nail Prediction  : 11.4 g/dL

Eye Weight       : 58.8%
Nail Weight      : 41.2%

Final Estimated Hb:
11.05 g/dL
```

---

# 5. Formula Umum

Secara matematis:

```text
w_eye =
(1 / MAE_eye)
/
[(1 / MAE_eye) + (1 / MAE_nail)]
```

```text
w_nail =
(1 / MAE_nail)
/
[(1 / MAE_eye) + (1 / MAE_nail)]
```

Kemudian:

```text
Hb_final =
(Hb_eye × w_eye)
+
(Hb_nail × w_nail)
```

---

# 6. Implementasi Python

```python
def combine_predictions(
    eye_hb,
    nail_hb,
    eye_mae,
    nail_mae
):
    raw_eye_weight = 1 / eye_mae
    raw_nail_weight = 1 / nail_mae

    total_weight = raw_eye_weight + raw_nail_weight

    eye_weight = raw_eye_weight / total_weight
    nail_weight = raw_nail_weight / total_weight

    final_hb = (
        eye_hb * eye_weight
        + nail_hb * nail_weight
    )

    return {
        "estimated_hb": round(final_hb, 2),
        "eye_weight": round(eye_weight, 3),
        "nail_weight": round(nail_weight, 3)
    }
```

Contoh pemakaian:

```python
result = combine_predictions(
    eye_hb=10.8,
    nail_hb=11.4,
    eye_mae=0.70,
    nail_mae=1.00
)

print(result)
```

Output:

```python
{
    "estimated_hb": 11.05,
    "eye_weight": 0.588,
    "nail_weight": 0.412
}
```

---

# 7. Integrasi ke FastAPI

File yang disarankan:

```text
backend/
│
├── services/
│   ├── inference.py
│   └── fusion.py
```

Isi `fusion.py`:

```python
def combine_predictions(
    eye_hb: float,
    nail_hb: float,
    eye_mae: float,
    nail_mae: float
):
    raw_eye_weight = 1 / eye_mae
    raw_nail_weight = 1 / nail_mae

    total_weight = raw_eye_weight + raw_nail_weight

    eye_weight = raw_eye_weight / total_weight
    nail_weight = raw_nail_weight / total_weight

    final_hb = (
        eye_hb * eye_weight
        + nail_hb * nail_weight
    )

    return {
        "estimated_hb": round(final_hb, 2),
        "eye_weight": round(eye_weight, 4),
        "nail_weight": round(nail_weight, 4)
    }
```

Di endpoint prediction:

```python
eye_result = predict_eye(eye_image)
nail_result = predict_nail(nail_image)

fusion_result = combine_predictions(
    eye_hb=eye_result["estimated_hb"],
    nail_hb=nail_result["estimated_hb"],
    eye_mae=EYE_MODEL_MAE,
    nail_mae=NAIL_MODEL_MAE
)
```

---

# 8. Contoh Response API

FastAPI dapat mengembalikan:

```json
{
  "success": true,
  "eye": {
    "estimated_hb": 10.8,
    "mae": 0.70
  },
  "nail": {
    "estimated_hb": 11.4,
    "mae": 1.00
  },
  "fusion": {
    "eye_weight": 0.588,
    "nail_weight": 0.412,
    "estimated_hb": 11.05
  }
}
```

Frontend React kemudian cukup menampilkan:

```text
Eye Estimate
10.8 g/dL

Nail Estimate
11.4 g/dL

Combined Estimate
11.05 g/dL
```

---

# 9. Perbandingan Metode Fusion

Untuk laporan lomba, bandingkan beberapa metode:

| Metode | Penjelasan |
|---|---|
| Eye Only | Hanya menggunakan model mata |
| Nail Only | Hanya menggunakan model kuku |
| Simple Average | Rata-rata 50:50 |
| Weighted Fusion | Bobot berdasarkan performa validasi |
| Stacking | Prediksi mata dan kuku menjadi input model ketiga |

Contoh tabel evaluasi:

| Metode | MAE | RMSE |
|---|---:|---:|
| Eye Model | 0.70 | 0.91 |
| Nail Model | 1.00 | 1.24 |
| Simple Average | 0.65 | 0.84 |
| Weighted Fusion | 0.58 | 0.76 |

Angka di atas hanya contoh.

Gunakan nilai hasil eksperimen aktual.

---

# 10. Baseline Simple Average

Sebagai baseline:

```text
Hb_final =
(Hb_eye + Hb_nail) / 2
```

Contoh:

```text
(10.8 + 11.4) / 2
= 11.1 g/dL
```

Metode ini mudah, tetapi menganggap kedua model sama akurat.

Gunakan ini hanya sebagai pembanding.

---

# 11. Advanced Option — Stacking

Jika dataset cukup besar, bisa dibuat model fusion tambahan.

Arsitektur:

```text
Eye Image
   ↓
Eye Model
   ↓
Eye Hb ───────────┐
                  │
                  ▼
              Fusion Model
                  │
                  ▼
             Final Hb
                  ▲
                  │
Nail Hb ──────────┘
   ↑
Nail Model
   ↑
Nail Image
```

Input model stacking:

```text
Eye Prediction
Nail Prediction
```

Model fusion bisa menggunakan:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

Contoh:

```python
X_fusion = [
    [eye_prediction, nail_prediction]
]

final_hb = fusion_model.predict(X_fusion)
```

---

# 12. Recommended Method for Anevia

Untuk tahap awal kompetisi, gunakan:

> **Weighted Late Fusion based on MAE**

Alasannya:

- sederhana
- mudah dijelaskan ke juri
- tidak membutuhkan model ketiga
- memanfaatkan performa aktual masing-masing model
- mudah diintegrasikan ke FastAPI
- lebih logis daripada bobot 50:50

Namun, metode fusion final harus tetap dievaluasi pada data validasi/test.

Jangan mengklaim weighted fusion lebih baik sebelum dibandingkan secara empiris.

---

# 13. Important Evaluation Rule

Bobot harus dihitung dari:

```text
validation dataset
```

atau dari skema evaluasi yang sudah ditentukan.

Jangan menentukan bobot menggunakan data test akhir jika data test digunakan sebagai evaluasi final.

Tujuannya agar tidak terjadi data leakage.

Ideal workflow:

```text
Training Data
    ↓
Train Eye Model
Train Nail Model

Validation Data
    ↓
Calculate Eye MAE
Calculate Nail MAE
    ↓
Determine Fusion Weights

Test Data
    ↓
Evaluate Final Fusion
```

---

# 14. Suggested Experiment for the Report

Evaluasi:

```text
1. Eye model MAE
2. Nail model MAE
3. Simple average MAE
4. Weighted fusion MAE
5. Compare RMSE
6. Compare R²
```

Metrics:

```text
MAE
RMSE
R²
```

Jika memungkinkan, tambahkan:

```text
Pearson Correlation
```

antara Hb hasil prediksi dan Hb aktual.

---

# 15. Final Anevia Architecture

```text
              ANEVIA

Eye Image
    │
    ▼
Eye Preprocessing
    │
    ▼
Eye Model
    │
    ▼
Eye Estimated Hb
    │
    │
    ├───────────────┐
                    │
                    ▼
             Weighted Fusion
                    │
                    ▼
            Final Estimated Hb
                    ▲
                    │
    ├───────────────┘
    │
Nail Estimated Hb
    ▲
    │
Nail Model
    ▲
    │
Nail Preprocessing
    ▲
    │
Nail Image
```

---

# 16. Key Formula for Documentation

Use this formula in the report:

```text
Hb_final = w_eye × Hb_eye + w_nail × Hb_nail
```

Where:

```text
w_eye + w_nail = 1
```

and:

```text
w_i ∝ 1 / MAE_i
```

This means the model with lower validation error receives a larger contribution to the final Hb estimate.

---

# 17. Notes

- Do not choose weights arbitrarily.
- Use validation performance.
- Keep test data separate.
- Compare fusion against single-model baselines.
- Use the actual Hb lab value as ground truth if available.
- Treat the final output as an **AI-estimated Hb value**, not a laboratory measurement.
- The final medical interpretation should be confirmed through proper clinical testing.
