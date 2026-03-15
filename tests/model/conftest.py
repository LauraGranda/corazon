"""Pytest fixtures for model evaluation tests."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification

N_SAMPLES: int = 100
N_FEATURES: int = 5


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic classification data for testing.

    Generates a binary classification dataset using sklearn's make_classification.
    Converts the numpy arrays to pandas DataFrame and Series for compatibility
    with the evaluation module.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Tuple of (X, y) where X is a DataFrame with 5 features and y is a Series
        with binary class labels.
    """
    X_arr, y_arr = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X: pd.DataFrame = pd.DataFrame(X_arr, columns=[f"feature_{i}" for i in range(N_FEATURES)])
    y: pd.Series = pd.Series(y_arr, name="target")
    return X, y
