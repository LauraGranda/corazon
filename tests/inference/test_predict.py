"""Unit tests for inference predict module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.inference.predict import (
    load_model,
    make_prediction_probabilities,
    make_predictions,
)


@pytest.fixture
def mock_model_path(tmp_path: Path) -> Path:
    """Create and save a dummy pipeline for testing."""
    model = Pipeline([("model", LogisticRegression())])
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)
    dump(model, tmp_path / "simple_logistic_regression.joblib")
    return tmp_path


def test_load_model_success(mock_model_path: Path) -> None:
    """Test successful model loading returns a Pipeline."""
    model = load_model(mock_model_path)
    assert isinstance(model, Pipeline)


def test_load_model_file_not_found(tmp_path: Path) -> None:
    """Test FileNotFoundError when model file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path, "nonexistent_model.joblib")


def test_make_predictions(mock_model_path: Path) -> None:
    """Test prediction returns ndarray with correct length."""
    model = load_model(mock_model_path)
    X = pd.DataFrame([[1, 2]], columns=["a", "b"])
    result = make_predictions(model, X)
    assert isinstance(result, np.ndarray)
    assert len(result) == 1


def test_make_prediction_probabilities(mock_model_path: Path) -> None:
    """Test probability prediction returns values between 0 and 1."""
    model = load_model(mock_model_path)
    X = pd.DataFrame([[1, 2]], columns=["a", "b"])
    result = make_prediction_probabilities(model, X)
    assert isinstance(result, np.ndarray)
    assert len(result) == 1
    assert 0 <= result[0] <= 1
