"""Batch inference pipeline for heart disease prediction."""

import importlib.util
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import using importlib after path is updated
predict_spec = importlib.util.spec_from_file_location(
    "predict",
    project_root / "inference" / "predict.py",
)
assert predict_spec is not None, "Failed to load predict module spec"
assert predict_spec.loader is not None, "Failed to load predict module loader"

predict = importlib.util.module_from_spec(predict_spec)
predict_spec.loader.exec_module(predict)

load_model = predict.load_model
make_predictions = predict.make_predictions

logger = logging.getLogger(__name__)


def run_batch_inference(
    input_data_path: Path,
    output_data_path: Path,
    model_dir: Path,
    model_name: str = "simple_logistic_regression.joblib",
) -> None:
    """Run batch inference on a dataset and save predictions to disk.

    Args:
        input_data_path: Path to input data file (.parquet or .csv).
        output_data_path: Path to save prediction results (.csv).
        model_dir: Directory containing the trained model.
        model_name: Name of the model file to load.
    """
    logger.info("Starting batch inference pipeline")

    # Load data
    if input_data_path.suffix == ".parquet":
        dataset = pd.read_parquet(input_data_path)
    elif input_data_path.suffix == ".csv":
        dataset = pd.read_csv(input_data_path)
    else:
        msg = f"Unsupported file format: {input_data_path.suffix}"
        raise ValueError(msg)

    logger.info("Loaded %d rows from %s", len(dataset), input_data_path)

    # Data preparation
    target = "disease"
    if target in dataset.columns:
        x_features = dataset.drop(target, axis="columns").copy()
    else:
        x_features = dataset.copy()

    # Remove rows with NaN to avoid sklearn preprocessing issues
    valid_mask = x_features.notna().all(axis=1)
    x_features_clean = x_features[valid_mask]
    logger.info("Removed %d rows with NaN values", len(x_features) - len(x_features_clean))

    # Load model and predict
    model = load_model(model_dir, model_name)
    logger.info("Model loaded from %s", model_dir / model_name)

    predictions = make_predictions(model, x_features_clean)
    logger.info("Generated %d predictions", len(predictions))

    # Save results
    results_df = dataset.copy()
    results_df["predicted_disease"] = None
    results_df.loc[valid_mask, "predicted_disease"] = predictions

    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_data_path, index=False)
    logger.info("Batch inference complete. Results saved to %s", output_data_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path.cwd().resolve()
    input_data = project_root / "data" / "03_primary" / "corazon_explored.parquet"
    output_data = project_root / "data" / "07_model_output" / "batch_predictions.csv"
    models_dir = project_root / "models"
    model_file = "simple_logistic_regression.joblib"

    run_batch_inference(input_data, output_data, models_dir, model_file)
