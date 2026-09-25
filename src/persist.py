"""Saving and loading fitted models with joblib."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def save_model(model: BaseEstimator, name: str, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name.lower()}.joblib"
    joblib.dump(model, path)
    logger.info("Saved model %s to %s", name, path)
    return path


def load_model(name: str, models_dir: Path) -> BaseEstimator:
    path = models_dir / f"{name.lower()}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}")
    return joblib.load(path)
