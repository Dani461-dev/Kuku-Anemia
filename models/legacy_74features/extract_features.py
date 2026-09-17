#!/usr/bin/env python3
"""
Feature Extraction Pipeline for Nail Hb Estimation
Uses HSV/LAB color space with physiologically meaningful statistics
"""

import cv2
import numpy as np
import pandas as pd
import ast
import json
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path("/mnt/d/Documents/Anemia Kuku/anemia-app/data")
PHOTO_DIR = DATA_DIR / "photo"
METADATA_CSV = DATA_DIR / "metadata.csv"
MODEL_DIR = Path("/mnt/d/Documents/Anemia Kuku/anemia-app/models")
MODEL_DIR.mkdir(exist_ok=True)

# WHO thresholds (g/dL)
WHO_THRESHOLD_FEMALE = 12.0
WHO_THRESHOLD_MALE = 13.0


# ============================================================
# ILLUMINATION NORMALIZATION
# ============================================================
def normalize_illumination(image_bgr: np.ndarray, skin_pixels: np.ndarray, method: str = "gray_world") -> np.ndarray:
    """
    Normalize image illumination using skin reference pixels.
    Assumes skin should have consistent color under canonical lighting.
    """
    if len(skin_pixels) == 0:
        return image_bgr
    
    # Estimate illuminant from skin pixels
    skin_mean = np.mean(skin_pixels, axis=0)  # BGR
    
    if method == "gray_world":
        # Gray world: average scene color should be gray
        # Scale each channel so skin mean becomes neutral
        target = np.array([180, 160, 140], dtype=np.float32)  # Typical skin BGR
        scale = target / (skin_mean + 1e-6)
        scale = np.clip(scale, 0.5, 2.0)  # Prevent extreme correction
    
    elif method == "white_patch":
        # White patch: brightest skin pixel = white
        skin_max = np.max(skin_pixels, axis=0)
        scale = 255.0 / (skin_max + 1e-6)
        scale = np.clip(scale, 0.5, 2.0)
    
    elif method == "shades_of_gray":
        # Shades of gray (Minkowski norm p=6)
        p = 6
        skin_norm = np.mean(skin_pixels ** p, axis=0) ** (1/p)
        target = np.array([180, 160, 140], dtype=np.float32)
        scale = target / (skin_norm + 1e-6)
        scale = np.clip(scale, 0.5, 2.0)
    
    else:
        return image_bgr
    
    # Apply scaling
    normalized = image_bgr.astype(np.float32)
    for c in range(3):
        normalized[:, :, c] *= scale[c]
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized


def mask_nail_region(image_bgr: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Extract nail pixels from bbox using GrabCut + color filtering.
    Removes background/skin contamination.
    """
    x1, y1, x2, y2 = bbox
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([]).reshape(0, 3)
    
    # GrabCut with rectangle slightly inset
    h, w = roi.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (5, 5, w-10, h-10)
    
    try:
        cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        gc_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    except:
        gc_mask = np.ones((h, w), np.uint8)
    
    # Additional color filtering: nail is typically less saturated than skin
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    # Keep pixels with moderate saturation (not background, not oversaturated)
    sat_mask = (saturation > 20) & (saturation < 200)
    
    combined_mask = gc_mask & sat_mask
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    pixels = roi[combined_mask > 0]
    return pixels


def get_nail_pixels(image: np.ndarray, nail_bboxes: List[List[int]]) -> np.ndarray:
    """Get nail pixels from all 3 bboxes with masking."""
    all_pixels = []
    for bbox in nail_bboxes:
        pixels = mask_nail_region(image, bbox)
        if len(pixels) > 0:
            all_pixels.append(pixels)
    if all_pixels:
        return np.vstack(all_pixels)
    return np.array([]).reshape(0, 3)


def get_skin_pixels(image: np.ndarray, skin_bboxes: List[List[int]]) -> np.ndarray:
    """Get skin pixels from all 3 bboxes (no masking needed, skin is uniform)."""
    all_pixels = []
    for bbox in skin_bboxes:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            all_pixels.append(crop.reshape(-1, 3))
    if all_pixels:
        return np.vstack(all_pixels)
    return np.array([]).reshape(0, 3)


# ============================================================
# COLOR SPACE FEATURE EXTRACTION (on normalized pixels)
# ============================================================
def extract_color_features(pixels_bgr: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Extract physiologically meaningful features from nail/skin pixels.
    
    HSV: Hue ~ hemoglobin oxygenation, Saturation ~ chroma, Value ~ brightness
    LAB: L* ~ lightness, a* ~ red-green (hemoglobin), b* ~ yellow-blue
    """
    if len(pixels_bgr) == 0:
        return {}
    
    # Convert to different color spaces
    pixels_rgb = pixels_bgr[:, [2, 1, 0]]  # BGR -> RGB
    pixels_hsv = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    pixels_lab = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    
    features = {}
    pre = f"{prefix}_" if prefix else ""
    
    # ===== Chromaticity (illumination-invariant) =====
    # rg-chromaticity: r = R/(R+G+B), g = G/(R+G+B)
    rgb_sum = pixels_rgb.sum(axis=1, keepdims=True) + 1e-6
    rg_chrom = pixels_rgb / rgb_sum
    for i, ch in enumerate(['r', 'g', 'b']):
        features[f'{pre}chrom_{ch}_mean'] = float(np.mean(rg_chrom[:, i]))
        features[f'{pre}chrom_{ch}_std'] = float(np.std(rg_chrom[:, i]))
        features[f'{pre}chrom_{ch}_median'] = float(np.median(rg_chrom[:, i]))
    
    # ===== HSV Statistics =====
    h, s, v = pixels_hsv[:, 0], pixels_hsv[:, 1], pixels_hsv[:, 2]
    for ch_name, ch_data in [('h', h), ('s', s), ('v', v)]:
        features[f'{pre}hsv_{ch_name}_mean'] = float(np.mean(ch_data))
        features[f'{pre}hsv_{ch_name}_std'] = float(np.std(ch_data))
        features[f'{pre}hsv_{ch_name}_median'] = float(np.median(ch_data))
    
    # Hue circular mean (important for hemoglobin hue)
    features[f'{pre}hsv_h_circular_mean'] = float(circular_mean_hue(h))
    
    # ===== LAB Statistics =====
    L, a, b_lab = pixels_lab[:, 0], pixels_lab[:, 1], pixels_lab[:, 2]
    for ch_name, ch_data in [('L', L), ('a', a), ('b', b_lab)]:
        features[f'{pre}lab_{ch_name.lower()}_mean'] = float(np.mean(ch_data))
        features[f'{pre}lab_{ch_name.lower()}_std'] = float(np.std(ch_data))
        features[f'{pre}lab_{ch_name.lower()}_median'] = float(np.median(ch_data))
    
    # LAB a* is most correlated with hemoglobin concentration
    features[f'{pre}lab_a_mean'] = float(np.mean(a))
    features[f'{pre}lab_a_median'] = float(np.median(a))
    
    return features


def skewness(data: np.ndarray) -> float:
    """Calculate sample skewness."""
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return 0.0
    return float(np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2)))


def circular_mean_hue(hues: np.ndarray) -> float:
    """Calculate circular mean of hue values (0-179 in OpenCV)."""
    radians = hues * (2 * np.pi / 180)
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    mean_angle = np.arctan2(sin_mean, cos_mean)
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return float(mean_angle * 180 / (2 * np.pi))


def extract_nail_skin_contrast(nail_pixels: np.ndarray, skin_pixels: np.ndarray) -> Dict[str, float]:
    """Extract contrast features between nail and skin regions (both normalized)."""
    if len(nail_pixels) == 0 or len(skin_pixels) == 0:
        return {}
    
    nail_lab = cv2.cvtColor(nail_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    skin_lab = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    
    features = {}
    
    # LAB contrast (nail - skin)
    for i, ch in enumerate(['L', 'a', 'b']):
        nail_mean = np.mean(nail_lab[:, i])
        skin_mean = np.mean(skin_lab[:, i])
        features[f'contrast_lab_{ch.lower()}_diff'] = float(nail_mean - skin_mean)
        features[f'contrast_lab_{ch.lower()}_ratio'] = float(nail_mean / (skin_mean + 1e-6))
    
    # HSV contrast
    nail_hsv = cv2.cvtColor(nail_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    skin_hsv = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    for i, ch in enumerate(['h', 's', 'v']):
        nail_mean = np.mean(nail_hsv[:, i])
        skin_mean = np.mean(skin_hsv[:, i])
        features[f'contrast_hsv_{ch}_diff'] = float(nail_mean - skin_mean)
        features[f'contrast_hsv_{ch}_ratio'] = float(nail_mean / (skin_mean + 1e-6))
    
    # Chromaticity contrast
    nail_rgb = nail_pixels[:, [2, 1, 0]]
    skin_rgb = skin_pixels[:, [2, 1, 0]]
    nail_chrom = nail_rgb / (nail_rgb.sum(axis=1, keepdims=True) + 1e-6)
    skin_chrom = skin_rgb / (skin_rgb.sum(axis=1, keepdims=True) + 1e-6)
    
    for i, ch in enumerate(['r', 'g', 'b']):
        features[f'contrast_chrom_{ch}_diff'] = float(np.mean(nail_chrom[:, i]) - np.mean(skin_chrom[:, i]))
        features[f'contrast_chrom_{ch}_ratio'] = float(np.mean(nail_chrom[:, i]) / (np.mean(skin_chrom[:, i]) + 1e-6))
    
    return features


# ============================================================
# BBOX PROCESSING
# ============================================================
def parse_bboxes(bbox_str: str) -> List[List[int]]:
    """Parse bbox string from metadata CSV."""
    try:
        return ast.literal_eval(bbox_str)
    except:
        return []


def crop_bbox(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Crop image to bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    return image[y1:y2, x1:x2]


def get_region_pixels(image: np.ndarray, bboxes: List[List[int]]) -> np.ndarray:
    """Combine pixels from multiple bboxes for a region (nail or skin)."""
    all_pixels = []
    for bbox in bboxes:
        crop = crop_bbox(image, bbox)
        if crop.size > 0:
            all_pixels.append(crop.reshape(-1, 3))
    if all_pixels:
        return np.vstack(all_pixels)
    return np.array([]).reshape(0, 3)


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================
def extract_features_from_image(image_path: Path, nail_bboxes: List, skin_bboxes: List) -> Dict:
    """Extract all features from a single image with illumination normalization."""
    image = cv2.imread(str(image_path))
    if image is None:
        return {}
    
    # Get raw skin pixels first (for illumination estimation)
    skin_pixels_raw = get_skin_pixels(image, skin_bboxes)
    
    # Normalize illumination using skin reference
    image_norm = normalize_illumination(image, skin_pixels_raw, method="gray_world")
    
    # Get masked nail pixels from normalized image
    nail_pixels = get_nail_pixels(image_norm, nail_bboxes)
    
    # Get skin pixels from normalized image
    skin_pixels = get_skin_pixels(image_norm, skin_bboxes)
    
    features = {}
    
    # Nail features (on normalized, masked pixels)
    nail_feats = extract_color_features(nail_pixels, prefix="nail")
    features.update(nail_feats)
    
    # Skin features (on normalized pixels)
    skin_feats = extract_color_features(skin_pixels, prefix="skin")
    features.update(skin_feats)
    
    # Nail-skin contrast features (both normalized)
    contrast_feats = extract_nail_skin_contrast(nail_pixels, skin_pixels)
    features.update(contrast_feats)
    
    return features


# ============================================================
# TRAINING PIPELINE
# ============================================================
def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load metadata, extract features from all images."""
    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded {len(df)} samples from metadata")
    
    X_list = []
    y_list = []
    feature_names = None
    failed = []
    
    for idx, row in df.iterrows():
        patient_id = row['PATIENT_ID']
        hb_level = row['HB_LEVEL_GperL']  # g/L
        image_path = PHOTO_DIR / f"{patient_id}.jpg"
        
        if not image_path.exists():
            failed.append(f"{patient_id}: image not found")
            continue
        
        nail_bboxes = parse_bboxes(row['NAIL_BOUNDING_BOXES'])
        skin_bboxes = parse_bboxes(row['SKIN_BOUNDING_BOXES'])
        
        features = extract_features_from_image(image_path, nail_bboxes, skin_bboxes)
        
        if not features:
            failed.append(f"{patient_id}: feature extraction failed")
            continue
        
        if feature_names is None:
            feature_names = sorted(features.keys())
        
        # Ensure consistent feature ordering
        feature_vector = [features.get(name, 0.0) for name in feature_names]
        X_list.append(feature_vector)
        y_list.append(hb_level / 10.0)  # Convert g/L to g/dL for modeling
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images")
    
    print(f"Successfully processed: {len(X_list)} samples")
    print(f"Failed: {len(failed)} samples")
    if failed:
        for f in failed[:5]:
            print(f"  - {f}")
    
    return np.array(X_list), np.array(y_list), feature_names


def train_model(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """Train ElasticNetCV with nested CV."""
    print(f"\nTraining on {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Hb range: {y.min():.1f} - {y.max():.1f} g/dL")
    print(f"Hb mean ± std: {y.mean():.1f} ± {y.std():.1f} g/dL")
    
    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ElasticNetCV with nested CV
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-4, 1, 50),
        cv=5,
        max_iter=10000,
        random_state=42,
        n_jobs=-1
    )
    
    # Nested CV for unbiased performance estimate
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=outer_cv, 
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_mae_std = cv_scores.std()
    
    print(f"\nNested CV MAE: {cv_mae:.3f} ± {cv_mae_std:.3f} g/dL")
    
    # Fit on full data
    model.fit(X_scaled, y)
    
    print(f"Best alpha: {model.alpha_:.6f}")
    print(f"Best l1_ratio: {model.l1_ratio_:.3f}")
    print(f"Non-zero coefficients: {np.sum(model.coef_ != 0)} / {len(model.coef_)}")
    
    # Feature importance
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nTop 20 Features by |Coefficient|:")
    print(coef_df.head(20).to_string(index=False))
    
    # Final evaluation on full data (optimistic)
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\nFull Data Performance (optimistic):")
    print(f"  MAE:  {mae:.3f} g/dL")
    print(f"  RMSE: {rmse:.3f} g/dL")
    print(f"  R²:   {r2:.3f}")
    
    # Clinical categorization accuracy
    print("\nClinical Categorization (WHO thresholds):")
    for gender, threshold in [('Female', WHO_THRESHOLD_FEMALE), ('Male', WHO_THRESHOLD_MALE)]:
        y_true_cat = (y < threshold).astype(int)
        y_pred_cat = (y_pred < threshold).astype(int)
        accuracy = (y_true_cat == y_pred_cat).mean()
        sensitivity = (y_pred_cat[y_true_cat == 1] == 1).mean() if (y_true_cat == 1).any() else 0
        specificity = (y_pred_cat[y_true_cat == 0] == 0).mean() if (y_true_cat == 0).any() else 0
        print(f"  {gender} (threshold {threshold} g/dL): Acc={accuracy:.2%}, Sens={sensitivity:.2%}, Spec={specificity:.2%}")
    
    return model, scaler, cv_mae, coef_df


def save_artifacts(model, scaler, feature_names, coef_df, cv_mae, X=None, y=None):
    """Save model, scaler, and metadata."""
    joblib.dump(model, MODEL_DIR / "elasticnet_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "robust_scaler.joblib")
    
    if X is not None:
        np.save(MODEL_DIR / "X_raw.npy", X)
    if y is not None:
        np.save(MODEL_DIR / "y_raw.npy", y)
    
    metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'cv_mae_g_dl': float(cv_mae),
        'best_alpha': float(model.alpha_),
        'best_l1_ratio': float(model.l1_ratio_),
        'n_nonzero_coef': int(np.sum(model.coef_ != 0)),
        'who_threshold_female_g_dl': WHO_THRESHOLD_FEMALE,
        'who_threshold_male_g_dl': WHO_THRESHOLD_MALE,
        'target_unit': 'g/dL'
    }
    
    with open(MODEL_DIR / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    coef_df.to_csv(MODEL_DIR / "feature_coefficients.csv", index=False)
    
    print(f"\nArtifacts saved to {MODEL_DIR}/")
    print("  - elasticnet_model.joblib")
    print("  - robust_scaler.joblib")
    print("  - model_metadata.json")
    print("  - feature_coefficients.csv")


# ============================================================
# INFERENCE FUNCTION
# ============================================================
def predict_hb(image_path: str, gender: str, model_path: str = None) -> Dict:
    """
    Inference function for production use.
    
    Args:
        image_path: Path to nail image
        gender: 'male' or 'female'
        model_path: Optional custom model path
    
    Returns:
        Dict with estimated_hb, category, threshold, etc.
    """
    if model_path is None:
        model_path = MODEL_DIR / "elasticnet_model.joblib"
    
    model = joblib.load(model_path)
    scaler = joblib.load(MODEL_DIR / "robust_scaler.joblib")
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    
    # For inference, we'd need MediaPipe to detect bboxes
    # This is a placeholder - actual implementation needs landmark detection
    # For now, assume bboxes are provided or use full image
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": "Image not found"}
    
    # TODO: Integrate MediaPipe Hands for automatic bbox detection
    # For now, return placeholder
    return {
        "success": True,
        "message": "Inference requires MediaPipe bbox detection - not yet implemented",
        "estimated_hb": None,
        "unit": "g/dL"
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NAIL HB ESTIMATION - FEATURE EXTRACTION & TRAINING")
    print("=" * 60)
    
    # Load data and extract features
    X, y, feature_names = load_dataset()
    
    if len(X) == 0:
        print("No data processed. Exiting.")
        exit(1)
    
    # Train model
    model, scaler, cv_mae, coef_df = train_model(X, y, feature_names)
    
    # Save artifacts
    save_artifacts(model, scaler, feature_names, coef_df, cv_mae, X, y)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)