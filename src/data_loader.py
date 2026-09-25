"""Loading and light cleaning of the banknote authentication dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def load_data(data_path: Path) -> pd.DataFrame:
    """Read the CSV from ``data_path`` and return it as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If expected columns are missing.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Place bill_authentication.csv in the data/ directory."
        )

    df = pd.read_csv(data_path)

    expected = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected column(s): {sorted(missing)}")

    logger.info("Loaded %d rows, %d columns from %s", *df.shape, data_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and rows with missing values.

    The banknote dataset is small and numeric-only, so cleaning is
    intentionally conservative: duplicates and nulls are the only issues
    worth handling automatically.
    """
    before = len(df)

    df = df.drop_duplicates()
    dropped_dupes = before - len(df)

    df = df.dropna()
    dropped_na = before - dropped_dupes - len(df)

    if dropped_dupes:
        logger.info("Dropped %d duplicate row(s)", dropped_dupes)
    if dropped_na:
        logger.info("Dropped %d row(s) with missing values", dropped_na)

    return df.reset_index(drop=True)


def load_and_clean(data_path: Path) -> pd.DataFrame:
    """Convenience wrapper: load then clean."""
    return clean_data(load_data(data_path))
