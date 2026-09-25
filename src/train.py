"""Pipeline orchestration: load -> EDA -> split -> train -> evaluate -> report."""

from __future__ import annotations

import logging

from src.config import PipelineConfig
from src.data_loader import load_and_clean
from src.eda import run_eda
from src.evaluate import (
    EvaluationResult,
    evaluate_model,
    plot_confusion_matrix,
    plot_model_comparison,
    results_to_dataframe,
    write_text_report,
)
from src.models import build_models
from src.persist import save_model
from src.preprocess import train_test_split_data
from src.visualize import plot_3d_scatter, plot_feature_importance

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig, skip_eda: bool = False) -> list[EvaluationResult]:
    """Run the full pipeline and return the per-model evaluation results."""
    config.ensure_output_dirs()

    logger.info("Loading data from %s", config.data_path)
    df = load_and_clean(config.data_path)

    if not skip_eda:
        logger.info("Running EDA")
        run_eda(df, config.feature_columns, config.target_column, config.figures_dir, config.reports_dir)
        plot_3d_scatter(
            df, config.feature_columns, config.target_column,
            config.figures_dir / "scatter_3d.png",
        )
    else:
        logger.info("Skipping EDA (skip_eda=True)")

    logger.info("Splitting data (test_size=%.2f, stratified)", config.test_size)
    split = train_test_split_data(
        df, config.feature_columns, config.target_column, config.test_size, config.random_state
    )

    models = build_models(random_state=config.random_state)
    results: list[EvaluationResult] = []

    class_names = [str(c) for c in sorted(df[config.target_column].unique())]

    for name, pipeline in models.items():
        logger.info("Training %s", name)
        result = evaluate_model(
            name, pipeline, split.X_train, split.y_train, split.X_test, split.y_test,
            cv_folds=config.cv_folds,
        )
        results.append(result)

        plot_confusion_matrix(result, config.figures_dir / f"confusion_{name.lower()}.png", class_names)
        plot_feature_importance(name, pipeline, config.feature_columns, config.figures_dir / f"importance_{name.lower()}.png")
        save_model(pipeline, name, config.models_dir)

    results_df = results_to_dataframe(results)
    results_df.to_csv(config.reports_dir / "model_comparison.csv")
    plot_model_comparison(results_df, config.figures_dir / "model_comparison.png")
    write_text_report(results, config.reports_dir / "evaluation_report.txt")

    logger.info("\n%s", results_df.to_string())
    best = results_df["test_accuracy"].idxmax()
    logger.info("Best model on test accuracy: %s (%.4f)", best, results_df.loc[best, "test_accuracy"])

    return results
