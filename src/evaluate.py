"""Model evaluation: metrics, reports, confusion-matrix plots, CV."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    name: str
    accuracy: float
    roc_auc: float
    cv_mean: float
    cv_std: float
    report: str
    confusion: np.ndarray

    def to_row(self) -> dict:
        return {
            "model": self.name,
            "test_accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "cv_accuracy_mean": self.cv_mean,
            "cv_accuracy_std": self.cv_std,
        }


def evaluate_model(
    name: str,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
) -> EvaluationResult:
    """Fit ``model`` on the training set and score it on the held-out test set.

    Also runs stratified k-fold cross-validation on the training set to get
    a less split-dependent estimate of generalisation accuracy.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_score)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_score)
    else:
        roc_auc = float("nan")

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")

    logger.info("%s: test accuracy=%.4f, cv accuracy=%.4f (+/- %.4f)", name, accuracy, cv_scores.mean(), cv_scores.std())

    return EvaluationResult(
        name=name,
        accuracy=accuracy,
        roc_auc=roc_auc,
        cv_mean=cv_scores.mean(),
        cv_std=cv_scores.std(),
        report=report,
        confusion=cm,
    )


def plot_confusion_matrix(result: EvaluationResult, out_path: Path, class_names: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=result.confusion, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {result.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix for %s to %s", result.name, out_path)


def results_to_dataframe(results: list[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results]).set_index("model")


def plot_model_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=results_df.index, y=results_df["test_accuracy"], ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Model comparison: SVM vs Random Forest vs XGBoost")
    for i, v in enumerate(results_df["test_accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved model comparison chart to %s", out_path)


def write_text_report(results: list[EvaluationResult], out_path: Path) -> None:
    lines = ["MODEL EVALUATION REPORT", "=" * 60, ""]
    for r in results:
        lines += [
            f"Model: {r.name}",
            "-" * 40,
            f"Test accuracy : {r.accuracy:.4f}",
            f"ROC AUC       : {r.roc_auc:.4f}",
            f"CV accuracy   : {r.cv_mean:.4f} (+/- {r.cv_std:.4f})",
            "",
            "Classification report:",
            r.report,
            "Confusion matrix:",
            np.array2string(r.confusion),
            "",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote evaluation report to %s", out_path)
