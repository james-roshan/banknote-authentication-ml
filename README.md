# Banknote Authentication — ML Pipeline

A modular, script-based rewrite of the original `PP1.ipynb` notebook, targeting
the real dataset in this project: `data/bill_authentication.csv` (banknote
authentication — predict genuine vs. forged from 4 numeric features).

## Project layout

```
PP1/
├── data/
│   └── bill_authentication.csv     # Variance, Skewness, Curtosis, Entropy, Class
├── src/
│   ├── config.py        # paths, column names, hyper-parameters (single source of truth)
│   ├── data_loader.py   # load_data / clean_data
│   ├── eda.py            # summary stats + all EDA plots, saved to outputs/figures
│   ├── preprocess.py     # stratified train/test split
│   ├── models.py         # SVM / Random Forest / XGBoost, each a scaler+model Pipeline
│   ├── evaluate.py       # accuracy, ROC-AUC, cross-validation, confusion matrix, reports
│   ├── visualize.py      # 3D scatter, feature importance plots
│   ├── persist.py        # save/load fitted models with joblib
│   └── train.py           # orchestrates the full pipeline
├── tests/
│   └── test_pipeline.py  # smoke tests (pytest)
├── outputs/
│   ├── figures/          # generated PNGs
│   ├── reports/          # text/CSV summaries
│   └── models/           # saved .joblib models
├── main.py                # CLI entry point
├── requirements.txt
└── plan.md                 # analysis of the original notebook vs. this dataset
```

## Setup

```powershell
# from the PP1/ directory
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python main.py                    # full run: EDA + train + evaluate SVM/RF/XGBoost
python main.py --skip-eda         # skip plots/summary, just train & evaluate
python main.py --test-size 0.2 --cv-folds 10 --random-state 0
python main.py --data path\to\other.csv   # same 4-feature + Class schema
```

All figures land in `outputs/figures/`, text/CSV reports in `outputs/reports/`,
and fitted models in `outputs/models/`.

## Test

```powershell
pytest
```

## What changed vs. the original notebook

- **Correct dataset.** The notebook loaded an unrelated customer-segmentation
  CSV from a hard-coded path on another machine. This project uses the real
  `bill_authentication.csv` with its actual `Class` target — no more predicting
  K-Means cluster IDs.
- **No notebook.** Logic lives in importable, testable modules under `src/`;
  `main.py` is the only thing you run.
- **Reproducible & scriptable.** Every step is a function with explicit
  inputs/outputs; the whole pipeline runs headlessly from the CLI (no
  `plt.show()` blocking, no re-running cells out of order).
- **Feature scaling** is applied consistently via `Pipeline(StandardScaler, ...)`
  for every model, and persisted together with the fitted estimator.
- **Stratified split + cross-validation** for a less split-dependent accuracy
  estimate, alongside the held-out test metrics.
- **Artifacts saved to disk** (figures, text reports, CSV comparison table,
  joblib models) instead of living only in notebook cell outputs.

See `plan.md` for the full analysis of the original notebook's issues.
