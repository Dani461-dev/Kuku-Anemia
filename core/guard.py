"""Plausibility guards for the seg26 runtime pipeline.

Prevents "liar" outputs: if detection is garbage, features drift out of the
training distribution, or the predicted Hb leaves a physiological range, the
caller can flag/retake instead of trusting the number.

Ranges (mirroring core/validate_mediapipe.py, adapted to the seg26 pipeline):
  - detection present, mask coverage of the nail box in [COV_MIN, COV_MAX]
  - >= FRAC_FEAT_OK fraction of the 42 features within the model's training
    percentile range [p1, p99]
  - predicted Hb within a physiological band
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core import config as cfg

# physiological band (g/dL)
HB_MIN_OK, HB_MAX_OK = 4.0, 18.0
# nail-mask coverage of the nail box (like validate_mediapipe: 15-99%), relaxed
COV_MIN_OK, COV_MAX_OK = 0.05, 0.99
FRAC_FEAT_OK = 0.70    # >= 70% features inside training range (reject severe drift)
DEFAULT_BOUNDS_CSV = cfg.OUTPUTS_DIR / "features_seg26_seg26_auto.csv"


def feature_bounds(source_csv=DEFAULT_BOUNDS_CSV) -> dict[str, tuple[float, float]]:
    """Per-feature [p1, p99] quantile range from the seg-runtime feature CSV."""
    import pandas as pd

    if not source_csv.exists():
        return {}
    df = pd.read_csv(source_csv)
    bounds: dict[str, tuple[float, float]] = {}
    for c in df.columns:
        if c.startswith(("NAIL_", "SKIN_")):
            col = df[c].astype(float).dropna()
            if len(col) == 0:
                continue
            bounds[c] = (float(np.percentile(col, 1)), float(np.percentile(col, 99)))
    return bounds


def feature_plausibility(vector, order, bounds: dict[str, tuple[float, float]]) -> tuple[bool, float]:
    """(ok, frac) — fraction of features falling inside training bounds."""
    if not bounds:
        return True, 1.0
    inside, total = 0, 0
    for name, value in zip(order, vector):
        lo, hi = bounds.get(name, (None, None))
        if lo is None:
            continue
        total += 1
        if lo <= float(value) <= hi:
            inside += 1
    frac = inside / max(1, total)
    return frac >= FRAC_FEAT_OK, frac


@dataclass
class GuardResult:
    ok: bool = True
    issues: list[str] = field(default_factory=list)
    feat_frac: float = 1.0

    @property
    def message(self) -> str:
        return "; ".join(self.issues)


def assess_hb(mask_coverage: float | None, hb_g_dl: float | None,
              vector, feature_order, bounds,
              n_instances: int = 0) -> GuardResult:
    """Evaluate a seg26 pipeline prediction; return flags.

    mask_coverage: nail-mask coverage of the nail box (0..1) or None if no nail.
    hb_g_dl: predicted Hb in g/dL (None if no prediction).
    vector: raw model feature vector (len == feature_order if hb present).
    """
    r = GuardResult()
    if n_instances <= 0:
        r.ok = False
        r.issues.append("no_nail_detected")
        return r
    if mask_coverage is None or not (COV_MIN_OK <= mask_coverage <= COV_MAX_OK):
        r.ok = False
        r.issues.append(f"mask_coverage({mask_coverage if mask_coverage is not None else 'NA'})")
    if hb_g_dl is None:
        r.ok = False
        r.issues.append("no_prediction")
        return r
    if not (HB_MIN_OK <= hb_g_dl <= HB_MAX_OK):
        r.ok = False
        r.issues.append(f"hb_out_of_range({hb_g_dl:.1f})")
    feat_ok, frac = feature_plausibility(vector, feature_order, bounds)
    r.feat_frac = frac
    if not feat_ok:
        r.ok = False
        r.issues.append(f"features_out_of_range({frac:.0%})")
    return r