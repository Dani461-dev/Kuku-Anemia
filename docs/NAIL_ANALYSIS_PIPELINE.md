# Nail Analysis Pipeline — Anevia

## Overview

Pipeline pemrosesan citra kuku untuk estimasi kadar hemoglobin (Hb) menggunakan pendekatan Computer Vision + Machine Learning tradisional.

---

## 1. MediaPipe Hands — Deteksi Landmark

**Input**: Citra kuku (RGB)  
**Output**: 21 landmark tangan (x, y, z normalized)

```python
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
landmarks = results.multi_hand_landmarks[0].landmark
```

**Landmark kunci untuk kuku**:
- `landmark[4]` — ujung ibu jari (thumb tip)
- `landmark[8]` — ujung telunjuk (index tip)
- `landmark[12]` — ujung jari tengah (middle tip)
- `landmark[16]` — ujung jari manis (ring tip)
- `landmark[20]` — ujung kelingking (pinky tip)

**Estimasi bounding box**:
- NAIL: area sekitar landmark tip (perlu kalibrasi offset)
- SKIN: area referensi di pangkal jari / punggung tangan

---

## 2. OpenCV Refine — Masking & Normalisasi Cahaya

### 2.1 Masking Piksel Kuku Murni

**Metode** (pilih salah satu / gabungan):

| Metode | Kelebihan | Kekurangan |
|--------|-----------|------------|
| **Otsu Thresholding** | Cepat, tidak butuh parameter | Sensitif pencahayaan tidak merata |
| **K-means Clustering (k=2/3)** | Bisa pisah kuku/background/kulit | Butuh inisialisasi, lebih lambat |
| **GrabCut** | Akurat batas objek | Butuh bounding box awal, iteratif |

**Rekomendasi**: Kombinasi — Otsu untuk cepat, GrabCut untuk refine batas.

```python
def refine_nail_mask(image, bbox, method="grabcut"):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    if method == "otsu":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif method == "kmeans":
        Z = roi.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # cluster dengan luminance tertinggi = kuku
        nail_cluster = np.argmax(centers.mean(axis=1))
        mask = (labels == nail_cluster).reshape(roi.shape[:2]).astype(np.uint8) * 255
    
    elif method == "grabcut":
        mask = np.zeros(roi.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (1, 1, w-2, h-2)
        cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask
```

### 2.2 Normalisasi Cahaya (Color Constancy)

**Pendekatan**: Gray World / White Patch / Color Checker (jika ada referensi)

```python
def normalize_illumination(image, mask, method="gray_world"):
    """Normalisasi pencahayaan menggunakan area kuku yang termasking."""
    masked_pixels = image[mask > 0]
    
    if method == "gray_world":
        # Asumsi: rata-rata warna scene = netral abu-abu
        mean_bgr = masked_pixels.mean(axis=0)
        scale = mean_bgr.mean() / mean_bgr
        normalized = cv2.convertScaleAbs(image, alpha=scale[0], beta=0)
        # apply per channel
        for c in range(3):
            image[:, :, c] = cv2.convertScaleAbs(image[:, :, c], alpha=scale[c])
    
    elif method == "white_patch":
        # Asumsi: piksel tercerah = putih
        max_bgr = masked_pixels.max(axis=0)
        scale = 255.0 / max_bgr
        for c in range(3):
            image[:, :, c] = cv2.convertScaleAbs(image[:, :, c], alpha=scale[c])
    
    return image
```

---

## 3. Ekstraksi Fitur Persentil RGB

**Input**: Citra ternormalisasi + mask kuku & mask kulit  
**Output**: 42 fitur (21 kuku + 21 kulit)

```python
def extract_percentile_features(image, nail_mask, skin_mask, percentiles=[5, 10, 25, 50, 75, 90, 95]):
    """
    Ekstraksi persentil RGB dari area kuku dan kulit.
    
    Total fitur: 7 percentiles × 3 channels × 2 regions = 42 features
    """
    features = {}
    
    for region_name, mask in [("nail", nail_mask), ("skin", skin_mask)]:
        pixels = image[mask > 0]  # (N, 3) BGR
        
        for p in percentiles:
            for c, ch_name in enumerate(["b", "g", "r"]):
                feat_name = f"{region_name}_{ch_name}_p{p}"
                features[feat_name] = np.percentile(pixels[:, c], p)
    
    return features  # dict 42 features
```

**Daftar fitur (42)**:
| Region | Channel | Percentiles |
|--------|---------|-------------|
| nail   | R, G, B | P5, P10, P25, P50, P75, P90, P95 |
| skin   | R, G, B | P5, P10, P25, P50, P75, P90, P95 |

---

## 4. Preprocessing & Modeling

### 4.1 RobustScaler

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)  # X shape: (n_samples, 42)
```

**Alasan RobustScaler**: Tahan terhadap outlier (pencahayaan ekstrem, artefak masking).

### 4.2 ElasticNetCV (Nested CV)

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold

# Outer CV: evaluasi performa
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV: hyperparameter tuning (ElasticNetCV built-in)
model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    alphas=np.logspace(-4, 1, 50),
    cv=5,  # inner CV
    max_iter=10000,
    random_state=42,
    n_jobs=-1
)

# Nested CV scoring
scores = cross_val_score(model, X_scaled, y, cv=outer_cv, scoring='neg_mean_absolute_error')
mae = -scores.mean()
```

**Hyperparameter**:
- `l1_ratio`: 0 (Ridge) → 1 (Lasso), elastic net di antaranya
- `alpha`: regularization strength
- **Nested CV** mencegah data leakage saat tuning

---

## 5. Prediksi Hb & Kategorisasi WHO

### 5.1 Prediksi

```python
def predict_hb(image, gender, model, scaler, hands):
    # 1. MediaPipe landmarks
    landmarks = detect_hand_landmarks(image, hands)
    
    # 2. Bounding box nail & skin
    nail_bbox, skin_bbox = estimate_bboxes(landmarks, image.shape)
    
    # 3. Masking & normalisasi
    nail_mask = refine_nail_mask(image, nail_bbox, method="grabcut")
    skin_mask = refine_skin_mask(image, skin_bbox)  # similar approach
    image_norm = normalize_illumination(image.copy(), nail_mask)
    
    # 4. Feature extraction
    features = extract_percentile_features(image_norm, nail_mask, skin_mask)
    X = np.array([list(features.values())])
    X_scaled = scaler.transform(X)
    
    # 5. Predict
    hb_pred = model.predict(X_scaled)[0]  # g/dL
    
    return hb_pred
```

### 5.2 Kategorisasi WHO (usia >15 tahun)

| Gender | Threshold Rendah | Kategori |
|--------|------------------|----------|
| Wanita | < 120 g/L (12.0 g/dL) | Anemia |
| Pria   | < 130 g/L (13.0 g/dL) | Anemia |

```python
def categorize_hb(hb_g_dl, gender):
    """
    hb_g_dl: float, estimasi Hb dalam g/dL
    gender: "male" atau "female"
    """
    threshold = 13.0 if gender.lower() in ["male", "pria", "laki-laki"] else 12.0
    
    if hb_g_dl < threshold:
        return "Rendah (Anemia)"
    else:
        return "Normal"
```

### 5.3 Response API

```json
{
  "success": true,
  "estimated_hb": 11.2,
  "unit": "g/dL",
  "gender": "female",
  "category": "Rendah (Anemia)",
  "threshold_used": 12.0,
  "features_used": 42,
  "model_info": {
    "type": "ElasticNetCV",
    "mae_cv": 0.85
  }
}
```

---

## File Structure Suggestion

```
backend/
├── services/
│   ├── nail_pipeline.py      # Steps 1-3
│   ├── features.py           # Step 3
│   ├── model.py              # Step 4 (load trained model + scaler)
│   └── inference.py          # Step 5 (orchestration)
├── models/
│   ├── elasticnet_model.joblib
│   └── robust_scaler.joblib
└── api/
    └── routes.py             # FastAPI endpoints
```

---

## Training Pipeline (Offline)

```python
# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load features (42 cols) + target Hb
df = pd.read_csv("features_dataset.csv")
X = df.drop("hb", axis=1).values
y = df["hb"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with nested CV
model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    alphas=np.logspace(-4, 1, 50),
    cv=5, max_iter=10000, random_state=42, n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"Best alpha: {model.alpha_:.6f}")
print(f"Best l1_ratio: {model.l1_ratio_:.3f}")

# Save
joblib.dump(model, "models/elasticnet_model.joblib")
joblib.dump(scaler, "models/robust_scaler.joblib")
```

---

## Notes

- **MediaPipe Hands** butuh lighting cukup baik; pertimbangkan augmentasi training
- **Masking** adalah bottleneck akurasi — validasi visual wajib
- **Normalisasi cahaya** kritis karena foto diambil user (bukan lab)
- **ElasticNet** otomatis feature selection via L1 component
- **Gender** wajib dikirim dari UI untuk kategorisasi WHO yang benar
- **Satuan**: model dilatih dalam g/dL, threshold WHO dalam g/L → konversi: 1 g/dL = 10 g/L