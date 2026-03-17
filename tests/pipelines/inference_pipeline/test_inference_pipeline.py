"""Unit tests for the batch inference pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.pipelines.inference_pipeline.inference_pipeline import run_batch_inference


@patch("src.pipelines.inference_pipeline.inference_pipeline.make_predictions")
@patch("src.pipelines.inference_pipeline.inference_pipeline.load_model")
def test_run_batch_inference(
    mock_load_model: MagicMock,
    mock_make_predictions: MagicMock,
    tmp_path: Path,
) -> None:
    """Test batch inference pipeline with mocked model and predictions."""
    # Setup input data
    input_file = tmp_path / "test_input.parquet"
    output_file = tmp_path / "output" / "predictions.csv"
    model_dir = tmp_path

    dummy_df = pd.DataFrame({"age": [50, 60], "disease": [1, 0]})
    dummy_df.to_parquet(input_file)

    # Configure mocks
    mock_load_model.return_value = MagicMock()
    mock_make_predictions.return_value = np.array([1, 0])

    # Run pipeline
    run_batch_inference(
        input_data_path=input_file,
        output_data_path=output_file,
        model_dir=model_dir,
        model_name="fake.joblib",
    )

    # Verify output
    assert output_file.exists()
    result = pd.read_csv(output_file)
    assert "predicted_disease" in result.columns
    assert result["predicted_disease"].tolist() == [1, 0]
