"""Unit tests for data loader functions."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import explore_data, load_data, save_data

# Constants for test fixtures
EXPECTED_SAMPLE_ROWS = 5
EXPECTED_COLUMNS = 14
FIRST_ROW_AGE = 63
MODIFIED_AGE = 100


class TestLoadData:
    """Tests for load_data function."""

    def test_load_data_returns_dataframe(self, sample_csv_path: Path) -> None:
        """Verify load_data returns a pandas DataFrame."""
        result = load_data(sample_csv_path)
        assert isinstance(result, pd.DataFrame)

    def test_load_data_correct_number_of_rows(self, sample_csv_path: Path) -> None:
        """Verify load_data loads all rows from CSV."""
        result = load_data(sample_csv_path)
        assert len(result) == EXPECTED_SAMPLE_ROWS

    def test_load_data_correct_number_of_columns(self, sample_csv_path: Path) -> None:
        """Verify load_data loads all columns from CSV."""
        result = load_data(sample_csv_path)
        assert len(result.columns) == EXPECTED_COLUMNS

    def test_load_data_contains_expected_columns(
        self, sample_csv_path: Path, expected_column_names: list
    ) -> None:
        """Verify load_data contains all expected column names."""
        result = load_data(sample_csv_path)
        assert list(result.columns) == expected_column_names

    def test_load_data_contains_age_column(self, sample_csv_path: Path) -> None:
        """Verify load_data contains 'age' column."""
        result = load_data(sample_csv_path)
        assert "age" in result.columns

    def test_load_data_contains_disease_column(self, sample_csv_path: Path) -> None:
        """Verify load_data contains 'disease' target column."""
        result = load_data(sample_csv_path)
        assert "disease" in result.columns

    def test_load_data_first_row_values(self, sample_csv_path: Path) -> None:
        """Verify load_data preserves first row values correctly."""
        result = load_data(sample_csv_path)
        assert result.loc[0, "age"] == FIRST_ROW_AGE
        assert result.loc[0, "sex"] == "Male"
        assert result.loc[0, "disease"] == 0

    def test_load_data_nonexistent_file_raises_error(self) -> None:
        """Verify load_data raises error when file does not exist."""
        nonexistent_path = Path("/nonexistent/path/to/file_xyz.csv")
        with pytest.raises(FileNotFoundError):
            load_data(nonexistent_path)

    def test_load_data_preserves_data_types(self, sample_csv_path: Path) -> None:
        """Verify load_data preserves correct data types."""
        result = load_data(sample_csv_path)
        assert result["age"].dtype == "int64"
        assert result["sex"].dtype == "object"
        assert result["disease"].dtype == "int64"

    def test_load_data_no_empty_dataframe(self, sample_csv_path: Path) -> None:
        """Verify load_data does not return empty DataFrame."""
        result = load_data(sample_csv_path)
        assert not result.empty


class TestExploreData:
    """Tests for explore_data function."""

    def test_explore_data_accepts_dataframe(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify explore_data accepts a DataFrame without error."""
        try:
            explore_data(sample_dataframe)
        except Exception as e:
            pytest.fail(f"explore_data raised {type(e).__name__}: {e}")

    def test_explore_data_with_empty_dataframe(self, empty_dataframe: pd.DataFrame) -> None:
        """Verify explore_data handles empty DataFrame gracefully."""
        try:
            explore_data(empty_dataframe)
        except Exception as e:
            pytest.fail(f"explore_data raised {type(e).__name__}: {e}")

    def test_explore_data_with_single_row_dataframe(
        self, single_row_dataframe: pd.DataFrame
    ) -> None:
        """Verify explore_data handles single-row DataFrame correctly."""
        try:
            explore_data(single_row_dataframe)
        except Exception as e:
            pytest.fail(f"explore_data raised {type(e).__name__}: {e}")

    def test_explore_data_with_large_dataframe(self, large_dataframe: pd.DataFrame) -> None:
        """Verify explore_data handles large DataFrame without error."""
        try:
            explore_data(large_dataframe)
        except Exception as e:
            pytest.fail(f"explore_data raised {type(e).__name__}: {e}")


class TestSaveData:
    """Tests for save_data function."""

    def test_save_data_creates_file(self, sample_dataframe: pd.DataFrame, tmp_path: Path) -> None:
        """Verify save_data creates a parquet file."""
        output_path = tmp_path / "output.parquet"
        save_data(sample_dataframe, output_path)
        assert output_path.exists()

    def test_save_data_creates_parquet_format(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Verify save_data creates file with .parquet extension."""
        output_path = tmp_path / "output.parquet"
        save_data(sample_dataframe, output_path)
        assert output_path.suffix == ".parquet"

    def test_save_data_preserves_dataframe_shape(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Verify save_data preserves DataFrame shape when saved."""
        output_path = tmp_path / "output.parquet"
        original_shape = sample_dataframe.shape
        save_data(sample_dataframe, output_path)
        loaded_df = pd.read_parquet(output_path)
        assert loaded_df.shape == original_shape

    def test_save_data_preserves_column_names(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Verify save_data preserves column names."""
        output_path = tmp_path / "output.parquet"
        save_data(sample_dataframe, output_path)
        loaded_df = pd.read_parquet(output_path)
        assert list(loaded_df.columns) == list(sample_dataframe.columns)

    def test_save_data_preserves_values(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Verify save_data preserves DataFrame values."""
        output_path = tmp_path / "output.parquet"
        save_data(sample_dataframe, output_path)
        loaded_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

    def test_save_data_creates_nested_directories(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Verify save_data creates nested directories if they don't exist."""
        output_path = tmp_path / "subdir1" / "subdir2" / "output.parquet"
        save_data(sample_dataframe, output_path)
        assert output_path.exists()

    def test_save_data_overwrites_existing_file(
        self, sample_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Verify save_data overwrites existing parquet file."""
        output_path = tmp_path / "output.parquet"
        save_data(sample_dataframe, output_path)

        modified_df = sample_dataframe.copy()
        modified_df.iloc[0, 0] = MODIFIED_AGE
        save_data(modified_df, output_path)

        loaded_df = pd.read_parquet(output_path)
        assert loaded_df.loc[0, "age"] == MODIFIED_AGE

    def test_save_data_with_empty_dataframe(
        self, empty_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Verify save_data handles empty DataFrame correctly."""
        output_path = tmp_path / "empty.parquet"
        save_data(empty_dataframe, output_path)
        loaded_df = pd.read_parquet(output_path)
        assert loaded_df.empty

    def test_save_data_with_single_row(
        self, single_row_dataframe: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Verify save_data preserves single-row DataFrame."""
        output_path = tmp_path / "single.parquet"
        save_data(single_row_dataframe, output_path)
        loaded_df = pd.read_parquet(output_path)
        assert len(loaded_df) == 1
