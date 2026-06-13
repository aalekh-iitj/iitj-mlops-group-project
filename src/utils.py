"""Common utilities: label mapping, metric helpers, env loading."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ID2LABEL_PATH = DATA_DIR / "id2label.json"
LABEL2ID_PATH = DATA_DIR / "label2id.json"
DATA_SUMMARY_PATH = DATA_DIR / "data_summary.json"


def load_id2label(path: Path = ID2LABEL_PATH) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_label2id(path: Path = LABEL2ID_PATH) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f).items()}


def save_id2label(mapping: Dict[int, str], path: Path = ID2LABEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(mapping.items())}, f, indent=2)


def save_label2id(mapping: Dict[str, int], path: Path = LABEL2ID_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in sorted(mapping.items(), key=lambda x: x[1])}, f, indent=2)


def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Read an env var, optionally required."""
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Add it to Kaggle Secrets (for training) or GitHub Actions secrets (for inference)."
        )
    return val
