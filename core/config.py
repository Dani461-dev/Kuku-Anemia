from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = CORE_DIR / "assets"
OUTPUTS_DIR = CORE_DIR / "outputs"
MODELS_DIR = CORE_DIR / "models"
CHART_DIR = CORE_DIR / "chart_output"

HAND_LANDMARKER_MODEL = ASSETS_DIR / "hand_landmarker.task"
CHART_REFS_JSON = CORE_DIR / "chart_refs.json"

PERCENTILE_LEVELS = [5, 15, 25, 50, 75, 85, 95]
COLORS = "RGB"

DATASET_WIDTH = 800
DATASET_HEIGHT = 600

# Fixed white reference region used by the original notebook:
# img[350:400, 300:350] i.e. rows(top,bottom), cols(left,right)
WHITE_ROWS = (350, 400)
WHITE_COLS = (300, 350)

# Hb target in grams per litre (g/L) as stored in metadata.csv
HB_MIN = 30.0
HB_MAX = 200.0

# Fingers we use for nail/skin detection (landmark ids of fingertips)
CANDIDATE_FINGERS = [8, 12, 16]  # index, middle, ring
ANATOMIC_MIDDLE_FINGER = 12

CLASS_NAMES = {0: "NAIL", 1: "SKIN"}

for d in (OUTPUTS_DIR, MODELS_DIR, CHART_DIR):
    d.mkdir(parents=True, exist_ok=True)