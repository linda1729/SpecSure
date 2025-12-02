import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATA_ROOT = BACKEND_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"
LABEL_DIR = DATA_ROOT / "labels"
PREDICTION_DIR = DATA_ROOT / "predictions"
PREVIEW_DIR = DATA_ROOT / "previews"
MODEL_DIR = DATA_ROOT / "models"
TMP_DIR = DATA_ROOT / "tmp"
META_PATH = DATA_ROOT / "meta.json"


def ensure_directories() -> None:
    """Create required data directories if they do not exist."""
    for path in [
        RAW_DIR,
        PREPROCESSED_DIR,
        LABEL_DIR,
        PREDICTION_DIR,
        PREVIEW_DIR,
        MODEL_DIR,
        TMP_DIR,
        META_PATH.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def default_meta() -> Dict[str, Any]:
    return {
        "datasets": [],
        "pipelines": [],
        "labels": [],
        "model_runs": [],
        "predictions": [],
        "evaluations": [],
    }


def read_meta() -> Dict[str, Any]:
    if META_PATH.exists():
        with META_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default_meta()


def write_meta(meta: Dict[str, Any]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
