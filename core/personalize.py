"""Per-user personalization for Hb estimation (Mannino Nat Commun 2018; PNAS 2025).

Prinsip (dari literatur): model populasi MAE ~1.3-1.5 g/dL; dengan personalisasi
(user punya 1+ hasil CBC lab), MAE pemantauan serial turun menjadi ~0.6-0.9 g/dL.

Cara kerja:
  - 0 titik kalibrasi -> keluaran model mentah (belum personalisasi).
  - 1 titik (CBC lab pertama) -> koreksi OFFSET:  raw + (hb_lab - raw_saat_lab).
    Menghapus bias sistematis per pengguna (tonus kulit, kamera, posisi jari).
  - >=2 titik (pemantauan serial) -> regresi LINEAR a*raw + b (least squares)
    atas pasangan (raw, lab); jatuh ke offset jika slope tidak stabil.

Profil per-user disimpan sebagai JSON kecil (path bebas, mis. per user_id).
Keluaran selalu di-clip ke rentang fisiologis [0.5, 25.0] g/dL.

Usage:
    from core.personalize import Personalizer
    p = Personalizer()
    raw = model.predict(img, boxes)["estimated_hb_g_dl"]   # 1.1 g/dL? -> 11.0
    p.add(raw_g_dl=11.0, actual_g_dl=9.8)                  # user pertama kali CBC
    cal = p.apply(raw_g_dl=10.5)                           # pemantauan berikutnya
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HB_MIN_G_DL, HB_MAX_G_DL = 0.5, 25.0


class Personalizer:
    """Kalibrasi per-pengguna untuk output model Hb (g/dL)."""

    def __init__(self, points: list[dict] | None = None):
        self.points: list[dict] = list(points or [])
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at

    # ------------------------------------------------------------------ state
    @property
    def is_calibrated(self) -> bool:
        return len(self.points) >= 1

    @property
    def n_points(self) -> int:
        return len(self.points)

    def add(self, raw_g_dl: float, actual_g_dl: float) -> None:
        """Tambahkan satu pasangan (prediksi mentah, nilai lab) ke profil."""
        raw_g_dl = float(np.clip(raw_g_dl, HB_MIN_G_DL, HB_MAX_G_DL))
        actual_g_dl = float(np.clip(actual_g_dl, HB_MIN_G_DL, HB_MAX_G_DL))
        self.points.append({"raw_g_dl": raw_g_dl, "actual_g_dl": actual_g_dl})
        self.updated_at = time.time()

    def reset(self) -> None:
        self.points.clear()
        self.updated_at = time.time()

    # ---------------------------------------------------------------- predict
    def apply(self, raw_g_dl: float) -> float:
        """Terapkan kalibrasi per-user ke prediksi mentah (g/dL)."""
        if not self.points:
            return float(np.clip(raw_g_dl, HB_MIN_G_DL, HB_MAX_G_DL))

        raw = np.array([p["raw_g_dl"] for p in self.points], dtype=float)
        lab = np.array([p["actual_g_dl"] for p in self.points], dtype=float)

        if len(self.points) == 1:
            out = raw_g_dl + (lab[0] - raw[0])          # offset 1-titik
        else:
            if np.var(raw) < 1e-6:                       # semua raw sama -> offset
                out = raw_g_dl + float((lab - raw).mean())
            else:
                a, b = np.polyfit(raw, lab, 1)           # linear >=2 titik
                out = a * raw_g_dl + b
        return float(np.clip(out, HB_MIN_G_DL, HB_MAX_G_DL))

    # ---------------------------------------------------------------- storage
    def to_dict(self) -> dict:
        return {
            "points": self.points,
            "n_points": len(self.points),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": "Per-user Hb personalization (offset 1pt / linear >=2pt)",
        }

    def save(self, path: str | Path) -> "Personalizer":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Personalizer":
        d = json.loads(Path(path).read_text())
        p = cls(points=d.get("points", []))
        p.created_at = d.get("created_at", p.created_at)
        p.updated_at = d.get("updated_at", p.updated_at)
        return p