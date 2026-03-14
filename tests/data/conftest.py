"""Pytest fixtures for data loader tests."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Create a temporary CSV file with sample heart disease data."""
    csv_path = tmp_path / "sample_data.csv"

    data = {
        "age": [63, 67, 37, 41, 56],
        "sex": ["Male", "Male", "Male", "Female", "Male"],
        "chest_pain": [
            "typical",
            "asymptomatic",
            "nonanginal",
            "nontypical",
            "nontypical",
        ],
        "rest_bp": [145, 160, 130, 130, 120],
        "chol": [233, 286, 250, 204, 236],
        "fbs": [1, 0, 0, 0, 0],
        "rest_ecg": [
            "left ventricular hypertrophy",
            "left ventricular hypertrophy",
            "normal",
            "left ventricular hypertrophy",
            "normal",
        ],
        "max_hr": [150, 108, 187, 172, 178],
        "exang": [0, 1, 0, 0, 0],
        "old_peak": [2.3, 1.5, 3.5, 1.4, 0.8],
        "slope": [3, 2, 3, 1, 1],
        "ca": [0.0, 3.0, 0.0, 0.0, 0.0],
        "thal": ["fixed", "normal", "normal", "normal", "normal"],
        "disease": [0, 1, 0, 0, 0],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def expected_column_names() -> list:
    """Return the expected column names for heart disease dataset."""
    return [
        "age",
        "sex",
        "chest_pain",
        "rest_bp",
        "chol",
        "fbs",
        "rest_ecg",
        "max_hr",
        "exang",
        "old_peak",
        "slope",
        "ca",
        "thal",
        "disease",
    ]


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample DataFrame with heart disease data."""
    return pd.DataFrame(
        {
            "age": [63, 67, 37],
            "sex": ["Male", "Male", "Male"],
            "chest_pain": ["typical", "asymptomatic", "nonanginal"],
            "rest_bp": [145, 160, 130],
            "chol": [233, 286, 250],
            "fbs": [1, 0, 0],
            "rest_ecg": [
                "left ventricular hypertrophy",
                "left ventricular hypertrophy",
                "normal",
            ],
            "max_hr": [150, 108, 187],
            "exang": [0, 1, 0],
            "old_peak": [2.3, 1.5, 3.5],
            "slope": [3, 2, 3],
            "ca": [0.0, 3.0, 0.0],
            "thal": ["fixed", "normal", "normal"],
            "disease": [0, 1, 0],
        }
    )


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with correct columns."""
    return pd.DataFrame(
        {
            "age": [],
            "sex": [],
            "chest_pain": [],
            "rest_bp": [],
            "chol": [],
            "fbs": [],
            "rest_ecg": [],
            "max_hr": [],
            "exang": [],
            "old_peak": [],
            "slope": [],
            "ca": [],
            "thal": [],
            "disease": [],
        }
    )


@pytest.fixture
def single_row_dataframe() -> pd.DataFrame:
    """Return a DataFrame with a single row."""
    return pd.DataFrame(
        {
            "age": [63],
            "sex": ["Male"],
            "chest_pain": ["typical"],
            "rest_bp": [145],
            "chol": [233],
            "fbs": [1],
            "rest_ecg": ["left ventricular hypertrophy"],
            "max_hr": [150],
            "exang": [0],
            "old_peak": [2.3],
            "slope": [3],
            "ca": [0.0],
            "thal": ["fixed"],
            "disease": [0],
        }
    )


@pytest.fixture
def large_dataframe() -> pd.DataFrame:
    """Return a DataFrame with 100 rows for large dataset testing."""
    return pd.DataFrame(
        {
            "age": [63 + i % 40 for i in range(100)],
            "sex": ["Male" if i % 2 == 0 else "Female" for i in range(100)],
            "chest_pain": ["typical", "asymptomatic", "nonanginal", "nontypical"] * 25,
            "rest_bp": [120 + i % 50 for i in range(100)],
            "chol": [200 + i % 100 for i in range(100)],
            "fbs": [i % 2 for i in range(100)],
            "rest_ecg": ["left ventricular hypertrophy", "normal"] * 50,
            "max_hr": [100 + i % 100 for i in range(100)],
            "exang": [i % 2 for i in range(100)],
            "old_peak": [float(i % 10) for i in range(100)],
            "slope": [i % 3 + 1 for i in range(100)],
            "ca": [float(i % 4) for i in range(100)],
            "thal": ["fixed", "normal", "reversible"] * 33 + ["fixed"],
            "disease": [i % 2 for i in range(100)],
        }
    )


@pytest.fixture
def messy_dataframe() -> pd.DataFrame:
    """Return a DataFrame with data quality issues for testing cleaning functions."""
    return pd.DataFrame(
        {
            "age": ["63", "abc", "45", "  58  ", "?", "50"],
            "sex": ["Male", "2345", "  Female  ", "Female", "Male", "NA"],
            "chest_pain": [
                "typical",
                "nontypical  ",
                "  asymptomatic",
                "garbage_val",
                "nonanginal",
                "?",
            ],
            "rest_bp": ["145", "160", "not_a_number", "130", "", "120"],
            "chol": ["233", "286", "250", "204", "null", "236"],
            "fbs": [1.0, 0.0, float("nan"), 0.0, 1.0, 0.0],
            "rest_ecg": [
                "normal",
                "left ventricular hypertrophy ",
                "5678",
                "normal",
                "ST-T wave abnormality",
                "",
            ],
            "max_hr": ["150", "108", "187", "xyz", "172", "178"],
            "exang": ["0", "1", "f", "0", "adfs", "1"],
            "old_peak": ["2.3", "1.5", "not_float", "1.4", "0.8", "None"],
            "slope": ["3", "afd", "2", "1", "3", "NULL"],
            "ca": ["0.0", "3.0", "bad", "0.0", "1.0", "?"],
            "thal": [
                "fixed",
                "87654",
                "  reversable  ",
                "normal",
                "reversable",
                "null",
            ],
            "disease": ["0", "1", "fsg", "0", "?", "1"],
        }
    )


@pytest.fixture
def pre_cast_dataframe() -> pd.DataFrame:
    """Return a DataFrame in pre-cast state (after standardize -> clean -> validate)."""
    return pd.DataFrame(
        {
            "age": [63.0, float("nan"), 45.0, 58.0],
            "sex": ["Male", None, "Female", "Female"],
            "chest_pain": ["typical", "nontypical", "asymptomatic", "nonanginal"],
            "rest_bp": [145.0, 160.0, float("nan"), 130.0],
            "chol": [233.0, 286.0, 250.0, float("nan")],
            "fbs": [1.0, 0.0, float("nan"), 0.0],
            "rest_ecg": ["normal", "left ventricular hypertrophy", None, "normal"],
            "max_hr": [150.0, 108.0, 187.0, float("nan")],
            "exang": ["0", "1", None, "0"],
            "old_peak": [2.3, 1.5, float("nan"), 1.4],
            "slope": ["3", "2", "3", None],
            "ca": ["0.0", "3.0", None, "0.0"],
            "thal": ["fixed", "normal", None, "normal"],
            "disease": ["0", "1", None, "0"],
        }
    )
