"""Central configuration for the banknote authentication pipeline.

All paths, column names and hyper-parameters used across the project are
defined here so every module (and the CLI) shares one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = the folder that contains this "src" package.
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"

DATA_PATH = DATA_DIR / "bill_authentication.csv"

FEATURE_COLUMNS = ["Variance", "Skewness", "Curtosis", "Entropy"]
TARGET_COLUMN = "Class"


@dataclass
class PipelineConfig:
    """Runtime settings for a single pipeline run."""

    data_path: Path = DATA_PATH
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    target_column: str = TARGET_COLUMN

    test_size: float = 0.3
    random_state: int = 42
    cv_folds: int = 5

    figures_dir: Path = FIGURES_DIR
    reports_dir: Path = REPORTS_DIR
    models_dir: Path = MODELS_DIR

    def ensure_output_dirs(self) -> None:
        for directory in (self.figures_dir, self.reports_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)
