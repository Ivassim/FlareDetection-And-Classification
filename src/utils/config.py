"""
Central configuration for Gas Flare Detection project.
All paths and hyper-parameters are defined here.
"""

from pathlib import Path

# ── Root ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # PFE-Object-Detection/

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_YAML    = ROOT / "Gas Flaring Detection.v15i.yolov8" / "data3.yaml"
TEST_IMAGES  = ROOT / "Gas Flaring Detection.v15i.yolov8" / "test3" / "images"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_VARIANT = "yolo11m.pt"
MODEL_RUN_NAME = "gas_flare_yolo11m_v1"

PREFERRED_BEST_WEIGHTS = ROOT / "outputs" / "models" / MODEL_RUN_NAME / "weights" / "best.pt"
PREFERRED_LAST_WEIGHTS = ROOT / "outputs" / "models" / MODEL_RUN_NAME / "weights" / "last.pt"

LEGACY_BEST_WEIGHTS = [
    ROOT / "outputs" / "models" / "gas_flare_yolov8m_v3" / "weights" / "best.pt",
    ROOT / "outputs" / "models" / "gas_flare_yolov8m_v2" / "weights" / "best.pt",
    ROOT / "outputs" / "models" / "gas_flare_yolov8s" / "weights" / "best.pt",
]
LEGACY_LAST_WEIGHTS = [
    ROOT / "outputs" / "models" / "gas_flare_yolov8m_v3" / "weights" / "last.pt",
    ROOT / "outputs" / "models" / "gas_flare_yolov8m_v2" / "weights" / "last.pt",
    ROOT / "outputs" / "models" / "gas_flare_yolov8s" / "weights" / "last.pt",
]
PRETRAINED_FALLBACKS = [
    ROOT / "yolo11m.pt",
    ROOT / "yolov8m.pt",
    ROOT / "yolov8s.pt",
]


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


BEST_WEIGHTS = _first_existing(
    [PREFERRED_BEST_WEIGHTS, *LEGACY_BEST_WEIGHTS, *PRETRAINED_FALLBACKS]
)
LAST_WEIGHTS = _first_existing([PREFERRED_LAST_WEIGHTS, *LEGACY_LAST_WEIGHTS])

# ── Output dirs ──────────────────────────────────────────────────────────────
OUTPUT_DIR      = ROOT / "outputs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
LOGS_DIR        = OUTPUT_DIR / "logs"

# ── Classes ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Dark-Flare",
    "Dark-Smoke",
    "Light-Flare",
    "Light-Smoke",
    "Medium-Flare",
    "Medium-Smoke",
]
NUM_CLASSES = len(CLASS_NAMES)

# ── Inference defaults ───────────────────────────────────────────────────────
IMG_SIZE   = 640
CONF_THRES = 0.25
IOU_THRES  = 0.45
DEVICE     = 0        # 0 = first GPU, "cpu" = CPU
