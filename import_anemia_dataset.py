# =============================================================================
#  import_anemia_dataset.py
#
#  Import dataset `sewa-rural-care/anemia-survey-dataset` dari Hugging Face
#  langsung ke drive, TANPA download penuh (~650 GB).
#
#  Trik utamanya: pyarrow column-pruning + remote range-read via HfFileSystem.
#  Hanya kolom yang dibutuhkan yang ditransfer per shard, jadi total transfer
#  hanya beberapa GB walau dataset aslinya 648 GB.
#
#  Yang diambil untuk SEMUA pasien:
#    - metadata  : patient_uuid, survey_*, cbc_*, hemoglobin_category
#    - image_conjunctiva        (mata)
#    - image_fingernails_open   (kuku terbuka)
#    - image_fingernails_closed (kuku mengepal)
#  Yang diambil HANYA 20 sampel:
#    - image_tongue (lidah)
#
#  RESUME-FRIENDLY:
#    - Metadata per-shard ditulis ke _meta_tmp/shard_XXXXX.csv, jadi shard yang
#      sudah selesai tidak dibaca ulang saat restart.
#    - Foto yang sudah ada di disk tidak ditulis ulang.
#
#  Output (disimpan langsung ke drive):
#    <OUT_DIR>/metadata.csv                -> 1 baris per pasien
#    <OUT_DIR>/images/conjunctiva/*.jpeg
#    <OUT_DIR>/images/fingernails_open/*.jpeg
#    <OUT_DIR>/images/fingernails_closed/*.jpeg
#    <OUT_DIR>/images/tongue/*.jpeg        -> maks 20 pasien
#
#  Jalankan:  python import_anemia_dataset.py
# =============================================================================

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "sewa-rural-care/anemia-survey-dataset"
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "anemia-dataset"
META_TMP = BASE_DIR / "_meta_tmp"
TONGUE_SAMPLE_LIMIT = 20          # jumlah sampel lidah yang diambil
N_WORKERS = 6                     # keseimbangan kecepatan vs rate-limit HF
SHARD_START, SHARD_END = 0, 736   # shard_00000 .. shard_00735

# Kolom yang dibutuhkan (semua metadata + 4 foto; PPG & camera_meta TIDAK diambil)
NEEDED_COLUMNS = (
    ["patient_uuid", "hemoglobin_category"]
    + [f"survey_{s}" for s in [
        "eligibility_comment", "eligible", "not_eligible_reason", "clinical_setting_type",
        "visit_reason", "service_attended", "smoking_history", "lack_of_energy",
        "breath_shortness", "fast_heart_beat", "losing_weight", "chest_pain", "dizziness",
        "headache", "leg_cramp", "brittle_nail_hair", "adult_other_symptom",
        "child_looks_pale", "child_losing_weight", "child_tired",
        "child_breathing_difficulty", "child_other_symptom", "data_collection_place",
        "data_collection_location", "data_collection_time", "data_collection_season",
        "light_intensity", "infant_height", "infant_weight", "systolic_bp",
        "diastolic_bp", "skin_tone_l", "skin_tone_b", "nail_polish_or_henna",
        "nail_examination", "other_nail_examination", "finger_or_hand_examination",
        "other_finger_or_hand_examination", "eye_examination", "other_eye_examination",
        "tongue_examination", "hemocue_venus_blood", "hemocue_capillary_blood",
        "sickle_cell_disease", "haemoglobin", "heart_rate", "hypertension",
        "diabetes_mellitus", "preeclampsia", "anemia", "disorders_of_blood",
        "type_of_blood_disorder", "cancer_or_malignancy", "other_known_diseases",
        "ambient_temperature", "body_temperature", "spo2", "clinical_examination",
        "device_modal", "age", "gender", "pregnancy_status",
    ]]
    + [f"cbc_{s}" for s in [
        "wbc_counts_10_3_ul", "rbc_counts_10_6_ul", "hgb_g_dl", "hct",
        "mcv_fl", "mch_pg", "mchc_g_dl", "platelet_counts_10_3_ul", "rdw_cv",
    ]]
    + ["image_conjunctiva", "image_fingernails_open",
       "image_fingernails_closed", "image_tongue"]
)

MODALITIES = {
    "image_conjunctiva": "conjunctiva",
    "image_fingernails_open": "fingernails_open",
    "image_fingernails_closed": "fingernails_closed",
    "image_tongue": "tongue",
}

_state = {
    "lock": __import__("threading").Lock(),
    "tongue_seen": 0,                  # di-set dari jumlah file lidah yang ada di disk
    "counts": {m: 0 for m in MODALITIES.values()},
}

fs = HfFileSystem()


def safe_name(uuid: str, orig_path: str) -> str:
    """Nama file: pakai nama asli dari parquet; fallback ke uuid.ext."""
    if orig_path and Path(orig_path).suffix.lower() in {".jpeg", ".jpg"}:
        return Path(orig_path).name
    return f"{uuid}.jpeg"


def init_state_from_disk():
    """Set tongue quota awal dari file yang sudah ada (untuk resume)."""
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    META_TMP.mkdir(parents=True, exist_ok=True)
    for folder in MODALITIES.values():
        d = OUT_DIR / "images" / folder
        if not d.exists():
            continue
        n = len(list(d.glob("*"))) if d.exists() else 0
        _state["counts"][folder] = n
    # batasi lidah persis ke TONGUE_SAMPLE_LIMIT (buang kelebihan hasil run lama)
    tdir = OUT_DIR / "images" / "tongue"
    if tdir.exists():
        existing = sorted(tdir.glob("*"))
        for f in existing[TONGUE_SAMPLE_LIMIT:]:
            f.unlink()
        _state["tongue_seen"] = min(len(existing), TONGUE_SAMPLE_LIMIT)
        _state["counts"]["tongue"] = min(len(existing), TONGUE_SAMPLE_LIMIT)
    print(f"[init] tongue_seen={_state['tongue_seen']} counts={dict(_state['counts'])}", flush=True)


def process_shard(idx: int) -> dict:
    """Baca satu shard (hanya kolom NEEDED), simpan foto & metadata per-shard."""
    shard = f"data/shard_{idx:05d}.parquet"
    path = f"hf://datasets/{REPO}/{shard}"
    meta_csv = META_TMP / f"shard_{idx:05d}.csv"

    if meta_csv.exists():
        return {"shard": shard, "ok": True, "skip": True, "rows": 0, "dt": 0.0}

    try:
        t0 = time.time()
        table = pq.read_table(path, filesystem=fs, columns=NEEDED_COLUMNS)
        rows = table.to_pylist()
    except Exception as exc:
        # shard placeholder (mis. shard_00000/00735 yg hanya 15 byte) -> skip
        if isinstance(exc, FileNotFoundError) or "not a parquet" in str(exc).lower() \
           or "magic" in str(exc).lower() or "Empty" in str(exc) or "EOF" in str(exc):
            return {"shard": shard, "ok": True, "skip": True, "rows": 0, "dt": 0.0}
        return {"shard": shard, "ok": False, "skip": False, "msg": str(exc)[:200]}

    for row in rows:
        uuid = row.get("patient_uuid")
        if not uuid:
            continue

        # --- foto ---
        for col, folder in MODALITIES.items():
            img = row.get(col)
            if not isinstance(img, dict) or not img.get("bytes"):
                continue
            if folder == "tongue":
                with _state["lock"]:
                    if _state["tongue_seen"] >= TONGUE_SAMPLE_LIMIT:
                        break          # kuota lidah habis -> berhenti untuk pasien ini
                    _state["tongue_seen"] += 1

            filepath = OUT_DIR / "images" / folder / safe_name(uuid, img.get("path", ""))
            filepath.parent.mkdir(parents=True, exist_ok=True)
            if not filepath.exists():
                filepath.write_bytes(img["bytes"])
            with _state["lock"]:
                _state["counts"][folder] += 1

        # --- metadata ---
        md = {k: v for k, v in row.items() if not k.startswith("image_")}
        for col, folder in MODALITIES.items():
            img = row.get(col)
            if isinstance(img, dict) and img.get("bytes"):
                md[f"image_{folder}"] = safe_name(uuid, img.get("path", ""))
            else:
                md[f"image_{folder}"] = None

        # tulis per-shard csv (resume-friendly)
        pd.DataFrame([md]).to_csv(meta_csv, index=False, mode="a",
                                  header=not meta_csv.exists())

    return {"shard": shard, "ok": True, "skip": False, "rows": len(rows),
            "dt": time.time() - t0}


def main():
    init_state_from_disk()
    print(f"Output dir : {OUT_DIR}", flush=True)
    print(f"Worker      : {N_WORKERS} threads", flush=True)

    shard_idxs = list(range(SHARD_START, SHARD_END))
    t_start = time.time()
    done = 0
    skipped = 0
    fails = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process_shard, i): i for i in shard_idxs}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if res.get("skip"):
                skipped += 1
            elif not res["ok"]:
                fails += 1
                print(f"  [ERROR] {res['shard']}: {res['msg']}", file=sys.stderr, flush=True)
            if done % 25 == 0 or done == len(shard_idxs):
                print(f"  progress {done}/{len(shard_idxs)} shards "
                      f"({time.time()-t_start:.0f}s) | counts={dict(_state['counts'])}",
                      flush=True)

    # --- gabungkan metadata per-shard -> metadata.csv ---
    parts = sorted(META_TMP.glob("shard_*.csv"))
    if parts:
        dfs = [pd.read_csv(p, dtype=str) for p in parts]
        df = pd.concat(dfs, ignore_index=True)
        out_csv = OUT_DIR / "metadata.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        n_meta = len(df)
    else:
        n_meta = 0
        out_csv = OUT_DIR / "metadata.csv"

    print()
    print(f"Selesai dalam {time.time()-t_start:.0f}s", flush=True)
    print(f"  shard OK      : {done - fails - skipped} / {len(shard_idxs)} (skip={skipped})", flush=True)
    print(f"  pasien        : {n_meta}", flush=True)
    if n_meta:
        def _dist(s):
            return df["hemoglobin_category"].value_counts(dropna=False).to_string()
        print("  distribusi label:")
        print(_dist(None), flush=True)
    print("  jumlah foto per modalitas:")
    for folder in MODALITIES.values():
        d = OUT_DIR / "images" / folder
        n = len(list(d.glob("*"))) if d.exists() else 0
        print(f"    {folder:20s}: {n}", flush=True)


if __name__ == "__main__":
    main()