"""Unit tests for the feature engineering pipeline module."""

import pandas as pd
from sklearn.compose import ColumnTransformer

from pipelines.feature_pipeline.build_features import create_preprocessor

# Constants for test assertions
EXPECTED_NUM_TRANSFORMERS = 3
EXPECTED_TRANSFORMER_NAMES = ["num", "cat", "ord"]
EXPECTED_NUM_FEATURES = 11  # 3 numeric + 2 chest_pain + 2 sex + 4 ordinal


class TestCreatePreprocessor:
    """Test suite for the create_preprocessor function."""

    def test_returns_column_transformer(self) -> None:
        """Test that create_preprocessor returns a ColumnTransformer instance."""
        preprocessor = create_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_three_transformers(self) -> None:
        """Test that the preprocessor has exactly three transformers."""
        preprocessor = create_preprocessor()
        assert len(preprocessor.transformers) == EXPECTED_NUM_TRANSFORMERS

    def test_transformer_names(self) -> None:
        """Test that transformer names are 'num', 'cat', and 'ord'."""
        preprocessor = create_preprocessor()
        names = [name for name, _, _ in preprocessor.transformers]
        assert names == EXPECTED_TRANSFORMER_NAMES

    def test_fit_transform_no_exception(self, dummy_df: pd.DataFrame) -> None:
        """Test that fit_transform runs without raising exceptions.

        This validates the entire pipeline structure and encoder wiring.

        Parameters
        ----------
        dummy_df : pd.DataFrame
            Dummy DataFrame with all required feature columns and some NaN values.
        """
        preprocessor = create_preprocessor()
        result = preprocessor.fit_transform(dummy_df)
        assert result is not None
        assert result.shape[0] == dummy_df.shape[0]

    def test_output_has_correct_shape(self, dummy_df: pd.DataFrame) -> None:
        """Test that the output has the expected number of columns.

        Checks that the transformed output has the correct number of features
        based on the combination of numeric, one-hot encoded categorical,
        and ordinal features. The exact number depends on the unique categories
        in the input data.

        Parameters
        ----------
        dummy_df : pd.DataFrame
            Dummy DataFrame with all required feature columns.
        """
        preprocessor = create_preprocessor()
        result = preprocessor.fit_transform(dummy_df)
        assert result.shape[1] == EXPECTED_NUM_FEATURES
