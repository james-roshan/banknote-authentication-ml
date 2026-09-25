"""Model definitions.

Every model is wrapped in a scikit-learn ``Pipeline`` with a
``StandardScaler`` step, so features are always scaled consistently and a
model can be persisted/loaded as a single object (scaler + estimator
together, no risk of applying the wrong scaler at inference time).
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return a name -> Pipeline mapping of the models used in this project."""
    return {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", gamma="scale", random_state=random_state)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),  # not required by trees, kept for a uniform pipeline shape
                ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state)),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
