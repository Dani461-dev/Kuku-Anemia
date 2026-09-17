"""Inference for one nail photo (doc §5.1) — predict Hb + WHO category.

Detection is deliberately decoupled: the caller supplies nail/skin boxes
(from GT boxes in offline tooling, or any detector: MediaPipe/YOLO at runtime).
This module is the single consumer of the trained model artifacts
(core/models/elasticnet_model.joblib + model_metadata.json).

Output unit follows the Anevia API convention: g/dL (model is fit in g/L).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import config as cfg
from core.categorize import categorize_hb
from core.detectors import DetectedFinger
from core.pipeline import load_rgb, patient_features
from core.features import feature_names

MODEL_PATH = cfg.MODELS_DIR / "elasticnet_model.joblib"
META_PATH = cfg.MODELS_DIR / "model_metadata.json"


class NailHbModel:
    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
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

    def predict(self, img_rgb: np.ndarray, boxes: DetectedFinger,
                gender: str = "female", white_source: str = "fixed", use_mask: bool = True) -> dict:
        """Predict Hb (g/dL) + WHO category. gender: 'male'/'female'.

        white_source: "fixed" cocok untuk layout dataset MSU (region img[350:400,300:350]).
        Untuk runtime foto tangan penuh gunakan "chart" (kartu ArUco, fallback auto).
        """
        X = self.features_for(img_rgb, boxes, white_source=white_source, use_mask=use_mask)
        hb_gperL = float(self.model.predict(X)[0])
        hb_g_dl = hb_gperL / 10.0
        return {
            "estimated_hb_g_dl": round(hb_g_dl, 2),
            "estimated_hb_gperL": round(hb_gperL, 1),
            "gender": gender,
            "category": categorize_hb(hb_g_dl, gender),
            "threshold_g_dl": self.meta["who_threshold_female_g_dl"]
            if gender not in ("male", "pria", "laki-laki", "laki", "m", "man")
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