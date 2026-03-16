"""Model evaluation module for classification tasks."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

SCORING_METRICS: list[str] = ["accuracy", "f1", "precision", "recall"]
RANDOM_STATE: int = 42


def evaluate_classification_model(
    model: Any,
    preprocessor: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 10,
) -> pd.DataFrame:
    """Evaluate a classification model using stratified cross-validation.

    Combines the given preprocessor and model into a scikit-learn Pipeline,
    then evaluates it using StratifiedKFold cross-validation across four
    standard classification metrics. Returns a single-row summary DataFrame
    with train score, CV mean, and CV std for each metric.

    Parameters
    ----------
    model : Any
        An unfitted scikit-learn-compatible classifier (e.g., LogisticRegression).
    preprocessor : Any
        An unfitted scikit-learn-compatible transformer (e.g., ColumnTransformer).
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector with binary class labels.
    cv_splits : int, optional
        Number of cross-validation folds, by default 10.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 12 columns (4 metrics x 3 stats):
        ``{metric}_train_score``, ``{metric}_cv_mean``, ``{metric}_cv_std``.
    """
    pipeline: Pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    cv: StratifiedKFold = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE
    )

    results: dict[str, Any] = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=SCORING_METRICS,
        return_train_score=True,
    )

    row: dict[str, float] = {}
    for metric in SCORING_METRICS:
        row[f"{metric}_train_score"] = float(np.mean(results[f"train_{metric}"]))
        row[f"{metric}_cv_mean"] = float(np.mean(results[f"test_{metric}"]))
        row[f"{metric}_cv_std"] = float(np.std(results[f"test_{metric}"]))

    return pd.DataFrame([row])
