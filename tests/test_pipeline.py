"""Smoke tests for the modular pipeline (not exhaustive, but catches breakage)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_loader import clean_data, load_data
from src.models import build_models
from src.preprocess import train_test_split_data


@pytest.fixture()
def config() -> PipelineConfig:
    return PipelineConfig()


def test_load_data(config: PipelineConfig) -> None:
    df = load_data(config.data_path)
    assert not df.empty
    assert set(config.feature_columns + [config.target_column]).issubset(df.columns)


def test_load_data_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


def test_clean_data_drops_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    cleaned = clean_data(df)
    assert len(cleaned) == 2


def test_train_test_split_is_stratified(config: PipelineConfig) -> None:
    df = load_data(config.data_path)
    df = clean_data(df)
    split = train_test_split_data(df, config.feature_columns, config.target_column, 0.3, 42)

    assert len(split.X_train) + len(split.X_test) == len(df)

    train_ratio = split.y_train.mean()
    test_ratio = split.y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_build_models_returns_expected_names() -> None:
    models = build_models()
    assert set(models.keys()) == {"SVM", "RandomForest", "XGBoost"}


def test_models_fit_predict_smoke(config: PipelineConfig) -> None:
    """Train each model on a tiny slice and make sure predict() runs end to end."""
    df = clean_data(load_data(config.data_path))
    split = train_test_split_data(df, config.feature_columns, config.target_column, 0.3, 42)

    models = build_models()
    for name, pipeline in models.items():
        pipeline.fit(split.X_train, split.y_train)
        preds = pipeline.predict(split.X_test)
        assert len(preds) == len(split.X_test)
