"""Unit tests for model evaluation module."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model.evaluation import evaluate_classification_model

EXPECTED_CV_SPLITS: int = 5
EXPECTED_METRICS: list[str] = ["accuracy", "f1", "precision", "recall"]
EXPECTED_NUM_COLUMNS: int = 12  # 4 metrics x 3 statistics


class TestEvaluateClassificationModel:
    """Test suite for the evaluate_classification_model function."""

    def test_returns_dataframe(self, classification_data: tuple) -> None:
        """Test that the function returns a pandas DataFrame.

        Parameters
        ----------
        classification_data : tuple
            Fixture providing (X, y) as (pd.DataFrame, pd.Series).
        """
        X, y = classification_data
        result = evaluate_classification_model(
            LogisticRegression(random_state=42, max_iter=1000),
            StandardScaler(),
            X,
            y,
            cv_splits=EXPECTED_CV_SPLITS,
        )
        assert isinstance(result, pd.DataFrame)

    def test_single_row(self, classification_data: tuple) -> None:
        """Test that the returned DataFrame has exactly one row.

        Parameters
        ----------
        classification_data : tuple
            Fixture providing (X, y) as (pd.DataFrame, pd.Series).
        """
        X, y = classification_data
        result = evaluate_classification_model(
            LogisticRegression(random_state=42, max_iter=1000),
            StandardScaler(),
            X,
            y,
            cv_splits=EXPECTED_CV_SPLITS,
        )
        assert result.shape[0] == 1

    def test_expected_columns(self, classification_data: tuple) -> None:
        """Test that the DataFrame has the expected number of columns.

        Each of the 4 metrics generates 3 columns (train_score, cv_mean, cv_std),
        resulting in 12 total columns.

        Parameters
        ----------
        classification_data : tuple
            Fixture providing (X, y) as (pd.DataFrame, pd.Series).
        """
        X, y = classification_data
        result = evaluate_classification_model(
            LogisticRegression(random_state=42, max_iter=1000),
            StandardScaler(),
            X,
            y,
            cv_splits=EXPECTED_CV_SPLITS,
        )
        assert result.shape[1] == EXPECTED_NUM_COLUMNS

    def test_metric_columns_present(self, classification_data: tuple) -> None:
        """Test that all expected metric columns are present in the result.

        For each metric, checks for the existence of:
        - {metric}_train_score
        - {metric}_cv_mean
        - {metric}_cv_std

        Parameters
        ----------
        classification_data : tuple
            Fixture providing (X, y) as (pd.DataFrame, pd.Series).
        """
        X, y = classification_data
        result = evaluate_classification_model(
            LogisticRegression(random_state=42, max_iter=1000),
            StandardScaler(),
            X,
            y,
            cv_splits=EXPECTED_CV_SPLITS,
        )
        for metric in EXPECTED_METRICS:
            assert f"{metric}_cv_mean" in result.columns
            assert f"{metric}_train_score" in result.columns
            assert f"{metric}_cv_std" in result.columns
