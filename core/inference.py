"""Inference for one nail photo (doc §5.1) — predict Hb + WHO category.

Detection is deliberately decoupled: the caller supplies nail/skin boxes
(from GT boxes in offline tooling, or any detector: MediaPipe/YOLO at runtime).
This module is the single consumer of the trained model artifacts
(core/models/elasticnet_model.joblib + model_metadata.json).

Output unit follows the Anevia API convention: g/dL (model is fit in g/L).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from core import config as cfg
from core.categorize import categorize_hb
from core.detectors import DetectedFinger
from core.pipeline import load_rgb, patient_features
from core.features import feature_names

MODEL_PATH = cfg.MODELS_DIR / "elasticnet_model.joblib"
META_PATH = cfg.MODELS_DIR / "model_metadata.json"
SEG_RUNTIME_MODEL = cfg.MODELS_DIR / "seg_runtime" / "elasticnet_model.joblib"
SEG_RUNTIME_META = cfg.MODELS_DIR / "seg_runtime" / "model_metadata.json"


def _model_paths() -> tuple[Path, Path]:
    """Resolve Hb model artifacts.

    Priority:
      1. env ANEVIA_HB_MODEL_DIR  (dir containing elasticnet_model.joblib)
      2. segment-runtime aligned model (core/models/seg_runtime) if it exists
      3. canonical model (core/models/)
    """
    env_dir = os.environ.get("ANEVIA_HB_MODEL_DIR")
    if env_dir:
        d = Path(env_dir)
        return d / "elasticnet_model.joblib", d / "model_metadata.json"
    if SEG_RUNTIME_MODEL.exists():
        return SEG_RUNTIME_MODEL, SEG_RUNTIME_META
    return MODEL_PATH, META_PATH


class NailHbModel:
    def __init__(self, model_path: Path | None = None, meta_path: Path | None = None):
        if model_path is None or meta_path is None:
            model_path, meta_path = _model_paths()
        import joblib

        self.model = joblib.load(model_path)
        self.meta = json.loads(Path(meta_path).read_text())
        self.feature_order = self.meta["feature_order"]

    def features_for(self, img_rgb: np.ndarray, boxes: DetectedFinger,
                     white_source: str = "fixed", use_mask: bool = True) -> np.ndarray:
        """Build the model feature vector (42 normalized features) for a finger."""
        feats = patient_features(img_rgb, boxes, white_source=white_source, use_mask=use_mask)
        feats = {k: v for k, v in feats.items() if not k.startswith("_")}
        vector = np.array([feats.get(c, 0.0) for c in self.feature_order], dtype=float)
        return vector.reshape(1, -1)

    def features_for_masks(self, img_rgb: np.ndarray, boxes: DetectedFinger,
                           nail_mask: np.ndarray | None = None,
                           skin_mask: np.ndarray | None = None,
                           white_source: str = "fixed") -> np.ndarray:
        """Feature vector where nail/skin pixels come from seg masks (full-frame)."""
        feats = patient_features(img_rgb, boxes, white_source=white_source,
                                 use_mask=True, nail_mask=nail_mask,
                                 skin_mask=skin_mask)
        feats = {k: v for k, v in feats.items() if not k.startswith("_")}
        vector = np.array([feats.get(c, 0.0) for c in self.feature_order], dtype=float)
        return vector.reshape(1, -1)

    def predict(self, img_rgb: np.ndarray, boxes: DetectedFinger,
                gender: str = "female", white_source: str = "fixed", use_mask: bool = True) -> dict:
        """Predict Hb (g/dL) + WHO category. gender: 'male'/'female'.

        white_source: "fixed" cocok untuk layout dataset MSU (region img[350:400,300:350]).
        Untuk runtime foto tangan penuh gunakan "chart" (kartu ArUco, fallback auto).
        """
        X = self.features_for(img_rgb, boxes, white_source=white_source, use_mask=use_mask)
        hb_gperL = float(self.model.predict(X)[0])
        hb_g_dl = hb_gperL / 10.0
        return self._wrap_result(hb_g_dl, hb_gperL, gender)

    def predict_masks(self, img_rgb: np.ndarray, boxes: DetectedFinger,
                      nail_mask: np.ndarray | None = None,
                      skin_mask: np.ndarray | None = None,
                      gender: str = "female", white_source: str = "fixed") -> dict:
        """Predict Hb using seg masks for nail/skin pixels."""
        X = self.features_for_masks(img_rgb, boxes, nail_mask=nail_mask,
                                    skin_mask=skin_mask, white_source=white_source)
        hb_gperL = float(self.model.predict(X)[0])
        hb_g_dl = hb_gperL / 10.0
        return self._wrap_result(hb_g_dl, hb_gperL, gender)

    def _wrap_result(self, hb_g_dl: float, hb_gperL: float, gender: str) -> dict:
        female_like = gender not in ("male", "pria", "laki-laki", "laki", "m", "man")
        return {
            "estimated_hb_g_dl": round(hb_g_dl, 2),
            "estimated_hb_gperL": round(hb_gperL, 1),
            "gender": gender,
            "category": categorize_hb(hb_g_dl, gender),
            "threshold_g_dl": self.meta["who_threshold_female_g_dl"] if female_like
            else self.meta["who_threshold_male_g_dl"],
            "model": self.meta.get("model", "RobustScaler + ElasticNet"),
            "cv_mae_g_dl": self.meta.get("cv_mae_g_dl"),
            "unit": "g/dL",
        }


def predict_hb(image_path: str | Path, nail_box, skin_box, gender: str = "female",
               white_source: str = "fixed", use_mask: bool = True) -> dict:
    """Convenience: load an image, supply a single finger's [t,l,b,r] boxes."""
    img = load_rgb(str(image_path))
    boxes = DetectedFinger(finger_id=12, nail_box=list(nail_box),
                           skin_box=list(skin_box), confidence=1.0, source="caller")
    return NailHbModel().predict(img, boxes, gender=gender,
                                 white_source=white_source, use_mask=use_mask)