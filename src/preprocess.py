"""Train/test splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_features_target(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_columns]
    y = df[target_column]
    return X, y


def train_test_split_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    test_size: float,
    random_state: int,
) -> SplitData:
    """Stratified train/test split so class balance is preserved in both sets."""
    X, y = split_features_target(df, feature_columns, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
