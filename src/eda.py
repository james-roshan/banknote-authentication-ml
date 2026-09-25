"""Exploratory data analysis: summary stats + plots saved to disk.

Every function here is a pure "save a figure / write a file" step so the
pipeline can run headlessly (no notebook, no blocking plt.show()).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: never blocks, works in CI/CLI

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def summarize(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> str:
    """Return a human-readable text summary of the dataset."""
    lines = [
        "DATASET SUMMARY",
        "=" * 60,
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "",
        "dtypes:",
        df.dtypes.to_string(),
        "",
        "Missing values per column:",
        df.isnull().sum().to_string(),
        "",
        "Descriptive statistics:",
        df[feature_columns].describe().to_string(),
        "",
        "Class balance:",
        df[target_column].value_counts().to_string(),
        "",
    ]
    return "\n".join(lines)


def save_summary(df: pd.DataFrame, feature_columns: list[str], target_column: str, out_path: Path) -> None:
    text = summarize(df, feature_columns, target_column)
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote data summary to %s", out_path)


def plot_boxplots(df: pd.DataFrame, feature_columns: list[str], out_path: Path) -> None:
    n = len(feature_columns)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.flatten()
    for ax, col in zip(axes, feature_columns):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved boxplots to %s", out_path)


def plot_histograms(df: pd.DataFrame, feature_columns: list[str], out_path: Path) -> None:
    n = len(feature_columns)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.flatten()
    for ax, col in zip(axes, feature_columns):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved histograms to %s", out_path)


def plot_class_balance(df: pd.DataFrame, target_column: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=target_column, data=df, ax=ax)
    ax.set_title("Class balance (0 = genuine, 1 = forged)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved class balance plot to %s", out_path)


def plot_correlation_heatmap(df: pd.DataFrame, feature_columns: list[str], target_column: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[feature_columns + [target_column]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s", out_path)


def plot_pairplot(df: pd.DataFrame, feature_columns: list[str], target_column: str, out_path: Path) -> None:
    grid = sns.pairplot(df, vars=feature_columns, hue=target_column, palette="Set1", diag_kind="hist")
    grid.figure.suptitle("Pairwise feature relationships by class", y=1.02)
    grid.savefig(out_path, dpi=150)
    plt.close(grid.figure)
    logger.info("Saved pairplot to %s", out_path)


def run_eda(df: pd.DataFrame, feature_columns: list[str], target_column: str, figures_dir: Path, reports_dir: Path) -> None:
    """Run the full EDA suite and save every artifact to disk."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    save_summary(df, feature_columns, target_column, reports_dir / "data_summary.txt")
    plot_boxplots(df, feature_columns, figures_dir / "boxplots.png")
    plot_histograms(df, feature_columns, figures_dir / "histograms.png")
    plot_class_balance(df, target_column, figures_dir / "class_balance.png")
    plot_correlation_heatmap(df, feature_columns, target_column, figures_dir / "correlation_heatmap.png")
    plot_pairplot(df, feature_columns, target_column, figures_dir / "pairplot.png")
