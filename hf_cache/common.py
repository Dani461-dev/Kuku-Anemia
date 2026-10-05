# =============================================================================
#  common.py — shared helpers for the parquet conversion pipeline
# =============================================================================

import json
import re
from pathlib import Path

import pandas as pd

MODALITIES = [
    "Anemia_Conjunctiva",
    "Anemia_Fingernails_Open",
    "Anemia_Fingernails_Closed",
    "Anemia_Tongue",
]
PPG_FOLDER = "PPG_Signals"

NULL_ALIASES = {"N/A", "n/a", "NA", "na", "null", "NULL", "None", "none", ""}
_DASH_PLACEHOLDER_RE = re.compile(r"^-+$")   # matches "-", "--", "----", etc.


def slugify(col: str) -> str:
    """'HGB (g/dL)' -> 'hgb_g_dl' ; 'Patient UUID' -> 'patient_uuid'"""
    col = col.strip().lower()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def clean_scalar(v):
    """
    Normalises a raw value to either None (if it's some flavor of missing —
    "N/A", empty, or a dash-placeholder like "----" seen in real CBC data)
    or the original value unchanged.
    """
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    if s in NULL_ALIASES or _DASH_PLACEHOLDER_RE.match(s):
        return None
    return v


def clean_scalar_to_str(v):
    """
    Like clean_scalar, but always returns None or a plain string. Used for
    every survey_*/cbc_* field so the final column type is unconditionally
    string across every patient/shard — this is what actually prevents
    pyarrow type-inference crashes when a batch mixes numeric-looking values
    ("11.7") with garbage placeholders ("----") in the same column.
    """
    v = clean_scalar(v)
    return None if v is None else str(v)


def haemoglobin_category(hb_value) -> str | None:
    """WHO thresholds (adult/female defaults) — same logic as original transformer.py"""
    try:
        hb = float(hb_value)
    except (TypeError, ValueError):
        return None
    if hb >= 12.0:
        return "normal"
    elif hb >= 11.0:
        return "mild_anemia"
    elif hb >= 8.0:
        return "moderate_anemia"
    else:
        return "severe_anemia"


# ─────────────────────────────────────────────────────────────────────────────
# METADATA LOADING — works from EITHER raw CSVs OR an already-built metadata.parquet
# ─────────────────────────────────────────────────────────────────────────────

def load_survey_cbc_from_csv(patient_dir: Path) -> dict:
    """
    Reads Survey_Details.csv + CBC_Report.csv directly (raw S3 format).
    dtype=str + keep_default_na=False: read every field as literal text,
    with no pandas type/NaN inference. This preserves the exact original
    text (e.g. "5.20" stays "5.20", not 5.2) and — more importantly — means
    every value is a plain string or an explicit "N/A"/"----"-style token
    we standardise ourselves, never a stray float/NaN that could poison
    type inference later when many patients are batched into one shard.
    """
    survey_df = pd.read_csv(
        patient_dir / "Survey_Data" / "Survey_Details.csv",
        dtype=str, keep_default_na=False,
    )
    survey_df.columns = survey_df.columns.str.strip()
    survey_row = survey_df.iloc[0].to_dict()

    cbc_df = pd.read_csv(
        patient_dir / "Survey_Data" / "CBC_Report.csv",
        dtype=str, keep_default_na=False,
    )
    cbc_df.columns = cbc_df.columns.str.strip()
    cbc_row = cbc_df.iloc[0].to_dict()
    cbc_row.pop("Patient UUID", None)

    return {"survey": survey_row, "cbc": cbc_row}


def load_survey_cbc_from_staged_parquet(patient_dir: Path) -> dict:
    """
    Reads an already-staged metadata.parquet (produced by the OLD pipeline's
    transformer.py). That file merged survey + cbc columns flat, and also
    added placeholder image_* path columns + hemoglobin_category, which we
    strip back out here since we rebuild those ourselves.
    """
    df = pd.read_parquet(patient_dir / "metadata.parquet")
    row = df.iloc[0].to_dict()

    drop_keys = {
        "image_conjunctiva", "image_fingernails_open",
        "image_fingernails_closed", "image_tongue",
        "image_ppg_signals_dir", "hemoglobin_category",
    }
    for k in drop_keys:
        row.pop(k, None)

    # We don't know which original keys were survey vs cbc anymore once merged,
    # so we just return everything under "survey" and let the caller re-split
    # using the known CBC column name list.
    cbc_cols = {
        "WBC Counts (10^3/uL)", "RBC Counts (10^6/uL)", "HGB (g/dL)",
        "HCT (%) ", "HCT (%)", "MCV (fL)", "MCH (pg)", "MCHC (g/dL)",
        "Platelet Counts (10^3/uL)", "RDW CV (%)",
    }
    survey_row = {k: v for k, v in row.items() if k not in cbc_cols}
    cbc_row = {k: v for k, v in row.items() if k in cbc_cols}
    return {"survey": survey_row, "cbc": cbc_row}


def build_flat_metadata_fields(merged: dict) -> dict:
    """
    Takes {"survey": {...}, "cbc": {...}} and returns flat, prefixed,
    null-standardised columns: survey_*, cbc_*, plus hemoglobin_category.
    Every survey_*/cbc_* value is None or a plain string — never a bare
    float/int — so the column type stays uniform across every patient and
    every shard regardless of what garbage placeholder values show up.
    """
    out = {}
    for col, val in merged["survey"].items():
        if slugify(col) == "patient_uuid":
            continue
        out[f"survey_{slugify(col)}"] = clean_scalar_to_str(val)
    for col, val in merged["cbc"].items():
        if slugify(col) == "patient_uuid":
            continue
        out[f"cbc_{slugify(col)}"] = clean_scalar_to_str(val)

    # NOTE: out["survey_haemoglobin"] is always present (possibly as None) —
    # dict.get()'s default only kicks in when the KEY is absent, not when its
    # value is None, so `out.get("survey_haemoglobin", out.get("cbc_hgb_g_dl"))`
    # would silently never fall through to the CBC value for any patient
    # missing a survey haemoglobin reading, even with a perfectly good CBC
    # value available. Use `or` so a None/falsy primary value actually falls
    # back.
    hb = out.get("survey_haemoglobin") or out.get("cbc_hgb_g_dl")
    out["hemoglobin_category"] = haemoglobin_category(hb)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TSV PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_modality_tsv(tsv_path: Path) -> dict:
    """
    Single-capture camera metadata TSV: columns key / resultValue / requestValue.
    Returns {key: {"result": ..., "request": ...}}
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for _, row in df.iterrows():
        out[row["key"]] = {
            "result": clean_scalar(row.get("resultValue")),
            "request": clean_scalar(row.get("requestValue")),
        }
    return out


def parse_ppg_tsv(tsv_path: Path) -> list:
    """
    Per-frame PPG sensor metadata TSV: one row per frame, wide columns.
    Returns a list of row-dicts, in frameNumber order.
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    if "frameNumber" in df.columns:
        df["_fn_sort"] = pd.to_numeric(df["frameNumber"], errors="coerce")
        df = df.sort_values("_fn_sort").drop(columns=["_fn_sort"])
    records = df.to_dict(orient="records")
    return [{k: clean_scalar(v) for k, v in r.items()} for r in records]


# ─────────────────────────────────────────────────────────────────────────────
# MEDIA FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def find_single_jpeg(modality_dir: Path) -> Path | None:
    jpegs = sorted(modality_dir.glob("*.jpeg")) + sorted(modality_dir.glob("*.jpg"))
    return jpegs[0] if jpegs else None


def find_single_tsv(modality_dir: Path) -> Path | None:
    tsvs = sorted(modality_dir.glob("*.tsv"))
    return tsvs[0] if tsvs else None


_FRAME_NUM_RE = re.compile(r"_(\d+)_v\d+s\d+\.jpe?g$", re.IGNORECASE)


def find_ppg_frames_sorted(ppg_dir: Path) -> list:
    """Returns PPG frame jpeg paths sorted numerically by frame index in filename."""
    frames = list(ppg_dir.glob("*.jpeg")) + list(ppg_dir.glob("*.jpg"))

    def key(p: Path):
        m = _FRAME_NUM_RE.search(p.name)
        return int(m.group(1)) if m else 10**9

    return sorted(frames, key=key)


def read_image_bytes(path: Path) -> dict:
    """HF `Image` feature-compatible dict for embedded storage."""
    with open(path, "rb") as f:
        data = f.read()
    return {"bytes": data, "path": path.name}


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL COLUMN ORDER — captured verbatim from the original Survey_Details.csv
# and CBC_Report.csv headers. Used by the reverse (parquet -> folders) script
# to rebuild CSVs with the exact original header text, not just slugs.
# ─────────────────────────────────────────────────────────────────────────────

ORIGINAL_SURVEY_COLUMNS = [
    "Eligibility Comment", "Eligible", "Not Eligible Reason", "Clinical Setting Type",
    "Visit Reason", "Service Attended", "Smoking History", "Lack Of Energy",
    "Breath Shortness", "Fast Heart Beat", "Losing Weight", "Chest Pain", "Dizziness",
    "Headache", "Leg Cramp", "Brittle Nail Hair", "Adult Other Symptom",
    "Child Looks Pale", "Child Losing Weight", "Child Tired",
    "Child Breathing Difficulty", "Child Other Symptom", "Data Collection Place",
    "Data Collection Location", "Data Collection Time", "Data Collection Season",
    "Light Intensity", "Infant Height", "Infant Weight", "Systolic Bp",
    "Diastolic Bp", "Skin Tone L", "Skin Tone B", "Nail Polish Or Henna",
    "Nail Examination", "Other Nail Examination", "Finger Or Hand Examination",
    "Other Finger Or Hand Examination", "Eye Examination", "Other Eye Examination",
    "Tongue Examination", "Hemocue(Venus Blood)", "Hemocue( Capillary blood)",
    "Sickle Cell Disease", "Haemoglobin", "Heart Rate", "Hypertension",
    "Diabetes Mellitus", "Preeclampsia", "Anemia", "Disorders Of Blood",
    "Type Of Blood Disorder", "Cancer Or Malignancy", "Other Known Diseases",
    "Ambient Temperature", "Body Temperature", "spo2", "Clinical Examination",
    "Device Modal", "Age", "Gender", "Pregnancy Status",
]

ORIGINAL_CBC_COLUMNS = [
    "WBC Counts (10^3/uL)", "RBC Counts (10^6/uL)", "HGB (g/dL)", "HCT (%)",
    "MCV (fL)", "MCH (pg)", "MCHC (g/dL)", "Platelet Counts (10^3/uL)",
    "RDW CV (%)",
]

# slug -> original header text, built once at import time
SURVEY_SLUG_TO_ORIGINAL = {slugify(c): c for c in ORIGINAL_SURVEY_COLUMNS}
CBC_SLUG_TO_ORIGINAL = {slugify(c): c for c in ORIGINAL_CBC_COLUMNS}

# Full, fixed list of every survey_*/cbc_* column name, in canonical order,
# plus hemoglobin_category. convert_to_shards.py builds every shard's arrow
# table using this EXACT list with an explicit pa.string() type per column —
# never through pandas dtype inference — so all ~750 shards share one
# identical schema no matter what garbage values ("----", stray floats,
# empty cells) show up in any individual patient's raw data.
METADATA_STRING_COLUMNS = (
    [f"survey_{slug}" for slug in SURVEY_SLUG_TO_ORIGINAL]
    + [f"cbc_{slug}" for slug in CBC_SLUG_TO_ORIGINAL]
    + ["hemoglobin_category"]
)


def unflatten_row_to_csvs(row: dict, uuid: str) -> tuple[dict, dict]:
    """
    Reverse of build_flat_metadata_fields(): takes one unified-parquet row
    and returns (survey_dict, cbc_dict) with ORIGINAL header text, ready to
    be written as Survey_Details.csv / CBC_Report.csv.
    """
    survey = {"Patient UUID": uuid}
    for slug, original in SURVEY_SLUG_TO_ORIGINAL.items():
        survey[original] = row.get(f"survey_{slug}")

    cbc = {"Patient UUID": uuid}
    for slug, original in CBC_SLUG_TO_ORIGINAL.items():
        cbc[original] = row.get(f"cbc_{slug}")

    return survey, cbc


def rebuild_modality_tsv_rows(camera_meta_json: str) -> list:
    """Reverse of parse_modality_tsv(): JSON string -> list of TSV rows."""
    meta = json.loads(camera_meta_json) if camera_meta_json else {}
    rows = []
    for key, vals in meta.items():
        rows.append({
            "key": key,
            "resultValue": vals.get("result"),
            "requestValue": vals.get("request"),
        })
    return rows
