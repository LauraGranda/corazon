"""Pytest fixtures for feature pipeline tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """Create a minimal dummy DataFrame with all expected feature columns.

    The DataFrame contains 3 rows with one NaN value per column to test
    the imputation logic. Categorical and ordinal columns use object dtype,
    and numeric columns use float dtype.

    Returns
    -------
    pd.DataFrame
        Dummy DataFrame with shape (3, 9) containing all feature columns:
        - Numeric: age, max_hr, old_peak
        - Categorical: chest_pain, sex
        - Ordinal: thal, slope, ca, exang
    """
    return pd.DataFrame(
        {
            "age": [50.0, np.nan, 60.0],
            "max_hr": [160.0, 150.0, np.nan],
            "old_peak": [1.5, np.nan, 2.0],
            "chest_pain": ["typical", np.nan, "asymptomatic"],
            "sex": ["Male", "Female", np.nan],
            "thal": ["normal", np.nan, "fixed"],
            "slope": ["1", "2", np.nan],
            "ca": ["0.0", np.nan, "1.0"],
            "exang": ["0", "1", np.nan],
        }
    )
