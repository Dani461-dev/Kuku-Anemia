"""WHO anemia classification helper (NAIL_ANALYSIS_PIPELINE.md §5.2).

Pure function — no I/O. Reused by the inference layer / API later.
Thresholds for adults (>15 years): women < 120 g/L, men < 130 g/L.
"""

from __future__ import annotations

WHO_FEMALE_GPERL = 120.0
WHO_MALE_GPERL = 130.0

GENDER_ALIASES_MALE = ("male", "pria", "laki-laki", "laki", "m", "man")


def threshold_g_dl(gender: str) -> float:
    """WHO anemia threshold in g/dL for the given gender."""
    g = (gender or "").strip().lower()
    if g in GENDER_ALIASES_MALE:
        return WHO_MALE_GPERL / 10.0
    return WHO_FEMALE_GPERL / 10.0


def threshold_gperL(gender: str) -> float:
    """WHO anemia threshold in g/L for the given gender."""
    return threshold_g_dl(gender) * 10.0


def categorize_hb(hb_g_dl: float, gender: str) -> str:
    """Classify an estimated Hb value against the WHO threshold.

    hb_g_dl: estimated Hb (can be float; NaN/None -> "Rendah (Anemia)"
    gender:  'male'/'female' (Indonesian aliases accepted)
    """
    if hb_g_dl is None or not (hb_g_dl == hb_g_dl):  # NaN check
        return "Tidak dapat diklasifikasikan"
    if hb_g_dl < threshold_g_dl(gender):
        return "Rendah (Anemia)"
    return "Normal"