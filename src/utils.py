"""Fungsi tambahan"""
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "results" / "metrics"
CM_DIR = ROOT / "results" / "confusion_matrix"
LOGS_DIR = ROOT / "results" / "logs"

MODELS = ["mobilenetv2", "efficientnetb0"]
PREPS = ["original", "minmax", "clahe"]

SCENARIOS = {
    ("mobilenetv2", "original"): "A",
    ("mobilenetv2", "minmax"): "B",
    ("mobilenetv2", "clahe"): "C",
    ("efficientnetb0", "original"): "D",
    ("efficientnetb0", "minmax"): "E",
    ("efficientnetb0", "clahe"): "F",
}


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Kunci seed Python, NumPy, dan TensorFlow (init bobot, shuffle, dropout, augmentasi)."""
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def run_name(model: str, prep: str, seed: int) -> str:
    return f"{model}_{prep}_seed{seed}"