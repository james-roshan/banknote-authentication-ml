"""Extra visualisations: 3D feature scatter and feature importance."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3d projection)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def plot_3d_scatter(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    out_path: Path,
) -> None:
    """3D scatter of the first three feature columns, coloured by class."""
    if len(feature_columns) < 3:
        logger.warning("Need at least 3 feature columns for a 3D scatter; skipping.")
        return

    x_col, y_col, z_col = feature_columns[:3]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df[x_col], df[y_col], df[z_col],
        c=df[target_column], cmap="viridis", s=40, alpha=0.7,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title("3D visualisation of banknote classes")
    legend = ax.legend(*scatter.legend_elements(), title=target_column)
    ax.add_artist(legend)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved 3D scatter to %s", out_path)


def plot_feature_importance(name: str, pipeline: Pipeline, feature_columns: list[str], out_path: Path) -> None:
    """Save a horizontal bar chart of feature importances, if the model exposes them."""
    clf = pipeline.named_steps.get("clf", pipeline)
    if not hasattr(clf, "feature_importances_"):
        logger.info("%s has no feature_importances_; skipping importance plot.", name)
        return

    importances = clf.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(feature_columns)[order], importances[order])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance - {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance plot for %s to %s", name, out_path)
