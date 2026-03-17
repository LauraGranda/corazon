"""Inference module for heart disease prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline


def load_model(
    model_dir: Path,
    model_name: str = "simple_logistic_regression.joblib",
) -> Pipeline:
    """Load trained pipeline from disk.

    Args:
        model_dir: Directory containing the model file.
        model_name: Name of the model file to load.

    Returns:
        Loaded sklearn Pipeline object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = model_dir / model_name
    if not model_path.exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)
    return load(model_path)


def make_predictions(model: Pipeline, input_data: pd.DataFrame) -> np.ndarray:
    """Return class predictions (0 or 1).

    Args:
        model: Fitted sklearn Pipeline.
        input_data: Input features as DataFrame.

    Returns:
        Array of predicted class labels (0 or 1).
    """
    predictions: np.ndarray = np.asarray(model.predict(input_data))
    return predictions


def make_prediction_probabilities(
    model: Pipeline,
    input_data: pd.DataFrame,
) -> np.ndarray:
    """Return probability of positive class (disease = 1).

    Args:
        model: Fitted sklearn Pipeline.
        input_data: Input features as DataFrame.

    Returns:
        Array of probabilities for the positive class.
    """
    probabilities: np.ndarray = np.asarray(model.predict_proba(input_data)[:, 1])
    return probabilities
