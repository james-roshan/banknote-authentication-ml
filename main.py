"""CLI entry point for the banknote authentication ML pipeline.

Usage:
    python main.py                     # full run: EDA + train + evaluate all models
    python main.py --skip-eda          # skip plots/summary, just train & evaluate
    python main.py --data path/to.csv  # use a different CSV (same column schema)
    python main.py --test-size 0.2 --cv-folds 10 --random-state 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import PipelineConfig
from src.train import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Banknote authentication ML pipeline")
    parser.add_argument("--data", type=Path, default=None, help="Path to the CSV dataset")
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction of data held out for testing")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA plots/summary and only train models")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    if args.data is not None:
        config.data_path = args.data

    run_pipeline(config, skip_eda=args.skip_eda)


if __name__ == "__main__":
    main()
