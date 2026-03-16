"""Unit tests for the training pipeline script."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import load
from sklearn.pipeline import Pipeline


def test_pipeline_script_exists() -> None:
    """Test that the training pipeline script file exists."""
    script_path = Path("src/pipelines/training_pipeline/train_pipeline.py")
    assert script_path.exists()


def test_model_file_generated() -> None:
    """Test that the trained model file exists after pipeline execution."""
    model_path = Path("models/simple_logistic_regression.joblib")
    assert model_path.exists()


@pytest.fixture
def trained_model() -> Pipeline:
    """Load the trained model for testing."""
    model_path = Path("models/simple_logistic_regression.joblib")
    return load(model_path)


def test_model_is_pipeline(trained_model: Pipeline) -> None:
    """Test that the loaded model is a Pipeline instance."""
    assert isinstance(trained_model, Pipeline)


def test_model_has_required_steps(trained_model: Pipeline) -> None:
    """Test that the pipeline has preprocessor and model steps."""
    step_names = [name for name, _ in trained_model.steps]
    assert "preprocessor" in step_names
    assert "model" in step_names


def test_model_can_predict(trained_model: Pipeline) -> None:
    """Test that the model can make predictions on sample data."""
    # Create sample data matching the expected feature structure
    sample_data = {
        "age": [50.0],
        "max_hr": [150.0],
        "old_peak": [2.5],
        "chest_pain": ["typical"],
        "sex": ["Male"],
        "thal": ["normal"],
        "slope": ["2"],
        "ca": ["0.0"],
        "exang": ["False"],
    }
    X_sample = pd.DataFrame(sample_data)
    predictions = trained_model.predict(X_sample)
    assert predictions is not None
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]


def test_model_can_predict_proba(trained_model: Pipeline) -> None:
    """Test that the model can return probability estimates."""
    sample_data = {
        "age": [50.0],
        "max_hr": [150.0],
        "old_peak": [2.5],
        "chest_pain": ["typical"],
        "sex": ["Male"],
        "thal": ["normal"],
        "slope": ["2"],
        "ca": ["0.0"],
        "exang": ["False"],
    }
    X_sample = pd.DataFrame(sample_data)
    proba = trained_model.predict_proba(X_sample)
    assert proba is not None
    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_file_is_readable() -> None:
    """Test that the model file can be loaded without errors."""
    model_path = Path("models/simple_logistic_regression.joblib")
    model = load(model_path)
    assert model is not None


def test_preprocessor_exists(trained_model: Pipeline) -> None:
    """Test that the pipeline has a fitted preprocessor."""
    preprocessor = trained_model.named_steps["preprocessor"]
    assert hasattr(preprocessor, "fit_transform")
    assert hasattr(preprocessor, "transformers_")
