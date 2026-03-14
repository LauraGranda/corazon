"""Unit tests for data cleaning and validation functions."""

import pandas as pd

from src.data.clean_data import (
    VALID_SEX,
    cast_types,
    clean_strings,
    invalidate_categorical,
    invalidate_numeric,
    standardize_nulls,
    validate_dataframe,
)

# Constants for test fixtures
MESSY_ROWS = 6
PRE_CAST_ROWS = 4
EXPECTED_COLUMNS = 14

# Constants for numeric test assertions
AGE_TEST_VALUE = 63.0
AGE_TEST_VALUE_2 = 67.0
AGE_TEST_VALUE_3 = 37.0
REST_BP_TEST_VALUE = 145.0


class TestStandardizeNulls:
    """Tests for standardize_nulls function."""

    def test_returns_dataframe(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify standardize_nulls returns a pandas DataFrame."""
        result = standardize_nulls(messy_dataframe)
        assert isinstance(result, pd.DataFrame)

    def test_question_mark_becomes_nan(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify '?' is replaced with NaN."""
        result = standardize_nulls(messy_dataframe)
        assert pd.isna(result.loc[4, "age"])

    def test_empty_string_becomes_nan(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify empty string '' is replaced with NaN."""
        result = standardize_nulls(messy_dataframe)
        assert pd.isna(result.loc[4, "rest_bp"])

    def test_null_string_becomes_nan(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify 'null' string is replaced with NaN."""
        result = standardize_nulls(messy_dataframe)
        assert pd.isna(result.loc[4, "chol"])

    def test_na_string_becomes_nan(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify 'NA' string is replaced with NaN."""
        result = standardize_nulls(messy_dataframe)
        assert pd.isna(result.loc[5, "sex"])

    def test_valid_value_not_replaced(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify valid values like 'Male' are not replaced with NaN."""
        result = standardize_nulls(messy_dataframe)
        assert result.loc[0, "sex"] == "Male"

    def test_does_not_modify_original(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify original DataFrame is not modified (immutability)."""
        original_sex_0 = messy_dataframe.loc[0, "sex"]
        standardize_nulls(messy_dataframe)
        assert messy_dataframe.loc[0, "sex"] == original_sex_0


class TestCleanStrings:
    """Tests for clean_strings function."""

    def test_returns_dataframe(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify clean_strings returns a pandas DataFrame."""
        df = standardize_nulls(messy_dataframe)
        result = clean_strings(df)
        assert isinstance(result, pd.DataFrame)

    def test_leading_spaces_stripped(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify leading spaces are stripped from string columns."""
        df = standardize_nulls(messy_dataframe)
        result = clean_strings(df)
        assert result.loc[3, "sex"] == "Female"
        assert result.loc[2, "chest_pain"] == "asymptomatic"

    def test_trailing_spaces_stripped(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify trailing spaces are stripped from string columns."""
        df = standardize_nulls(messy_dataframe)
        result = clean_strings(df)
        assert result.loc[1, "chest_pain"] == "nontypical"
        assert result.loc[1, "rest_ecg"] == "left ventricular hypertrophy"

    def test_non_object_column_unchanged(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify numeric columns like fbs are not affected."""
        df = standardize_nulls(messy_dataframe)
        result = clean_strings(df)
        assert result["fbs"].dtype == "float64"

    def test_does_not_modify_original(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify original DataFrame is not modified (immutability)."""
        df = standardize_nulls(messy_dataframe)
        original_sex_2 = df.loc[2, "sex"]
        clean_strings(df)
        assert df.loc[2, "sex"] == original_sex_2


class TestInvalidateCategorical:
    """Tests for invalidate_categorical function."""

    def test_returns_series(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify invalidate_categorical returns a pandas Series."""
        result = invalidate_categorical(messy_dataframe["sex"], VALID_SEX)
        assert isinstance(result, pd.Series)

    def test_valid_value_preserved(self) -> None:
        """Verify valid categorical values are preserved."""
        series = pd.Series(["Male", "Female", "Male"])
        result = invalidate_categorical(series, VALID_SEX)
        assert result.iloc[0] == "Male"
        assert result.iloc[1] == "Female"

    def test_invalid_value_becomes_nan(self) -> None:
        """Verify invalid values become NaN."""
        series = pd.Series(["Male", "2345", "Female"])
        result = invalidate_categorical(series, VALID_SEX)
        assert pd.isna(result.iloc[1])

    def test_existing_nan_remains_nan(self) -> None:
        """Verify existing NaN values are preserved as NaN."""
        series = pd.Series(["Male", None, "Female"])
        result = invalidate_categorical(series, VALID_SEX)
        assert pd.isna(result.iloc[1])

    def test_all_invalid_returns_all_nan(self) -> None:
        """Verify series with all invalid values becomes all NaN."""
        series = pd.Series(["garbage1", "garbage2", "garbage3"])
        result = invalidate_categorical(series, VALID_SEX)
        assert result.isna().all()


class TestInvalidateNumeric:
    """Tests for invalidate_numeric function."""

    def test_returns_series(self) -> None:
        """Verify invalidate_numeric returns a pandas Series."""
        series = pd.Series(["63", "67", "37"])
        result = invalidate_numeric(series)
        assert isinstance(result, pd.Series)

    def test_numeric_string_converts_to_float(self) -> None:
        """Verify numeric strings are converted to float."""
        series = pd.Series(["63", "67", "37"])
        result = invalidate_numeric(series)
        assert result.iloc[0] == AGE_TEST_VALUE
        assert result.iloc[1] == AGE_TEST_VALUE_2

    def test_non_numeric_string_becomes_nan(self) -> None:
        """Verify non-numeric strings become NaN."""
        series = pd.Series(["63", "abc", "37"])
        result = invalidate_numeric(series)
        assert result.iloc[0] == AGE_TEST_VALUE
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == AGE_TEST_VALUE_3

    def test_existing_nan_remains_nan(self) -> None:
        """Verify existing NaN values are preserved as NaN."""
        series = pd.Series(["63", float("nan"), "37"])
        result = invalidate_numeric(series)
        assert pd.isna(result.iloc[1])

    def test_result_dtype_is_float_or_int(self) -> None:
        """Verify result is numeric dtype (float64 or int64)."""
        series = pd.Series(["63", "67", "37"])
        result = invalidate_numeric(series)
        assert result.dtype in ["float64", "int64"]


class TestValidateDataframe:
    """Tests for validate_dataframe function."""

    def test_returns_dataframe(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify validate_dataframe returns a pandas DataFrame."""
        df = standardize_nulls(messy_dataframe)
        df = clean_strings(df)
        result = validate_dataframe(df)
        assert isinstance(result, pd.DataFrame)

    def test_categorical_invalid_becomes_nan(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify invalid categorical values become NaN."""
        df = standardize_nulls(messy_dataframe)
        df = clean_strings(df)
        result = validate_dataframe(df)
        assert pd.isna(result.loc[1, "sex"])
        assert pd.isna(result.loc[3, "chest_pain"])

    def test_numeric_invalid_becomes_nan(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify invalid numeric values become NaN."""
        df = standardize_nulls(messy_dataframe)
        df = clean_strings(df)
        result = validate_dataframe(df)
        assert pd.isna(result.loc[1, "age"])
        assert pd.isna(result.loc[2, "rest_bp"])

    def test_valid_categorical_preserved(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify valid categorical values are preserved."""
        df = standardize_nulls(messy_dataframe)
        df = clean_strings(df)
        result = validate_dataframe(df)
        assert result.loc[0, "sex"] == "Male"
        assert result.loc[0, "chest_pain"] == "typical"

    def test_valid_numeric_preserved(self, messy_dataframe: pd.DataFrame) -> None:
        """Verify valid numeric values are converted to float."""
        df = standardize_nulls(messy_dataframe)
        df = clean_strings(df)
        result = validate_dataframe(df)
        assert result.loc[0, "age"] == AGE_TEST_VALUE
        assert result.loc[0, "rest_bp"] == REST_BP_TEST_VALUE


class TestCastTypes:
    """Tests for cast_types function."""

    def test_returns_dataframe(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify cast_types returns a pandas DataFrame."""
        result = cast_types(pre_cast_dataframe)
        assert isinstance(result, pd.DataFrame)

    def test_age_dtype_is_nullable_integer(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify age is cast to Int64 (nullable integer)."""
        result = cast_types(pre_cast_dataframe)
        assert result["age"].dtype == "Int64"

    def test_rest_bp_dtype_is_nullable_integer(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify rest_bp is cast to Int64 (nullable integer)."""
        result = cast_types(pre_cast_dataframe)
        assert result["rest_bp"].dtype == "Int64"

    def test_chol_dtype_is_nullable_integer(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify chol is cast to Int64 (nullable integer)."""
        result = cast_types(pre_cast_dataframe)
        assert result["chol"].dtype == "Int64"

    def test_max_hr_dtype_is_nullable_integer(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify max_hr is cast to Int64 (nullable integer)."""
        result = cast_types(pre_cast_dataframe)
        assert result["max_hr"].dtype == "Int64"

    def test_old_peak_dtype_is_float64(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify old_peak is cast to float64."""
        result = cast_types(pre_cast_dataframe)
        assert result["old_peak"].dtype == "float64"

    def test_fbs_dtype_is_boolean(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify fbs is cast to boolean (nullable)."""
        result = cast_types(pre_cast_dataframe)
        assert result["fbs"].dtype == "boolean"

    def test_exang_dtype_is_boolean(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify exang is cast to boolean (nullable)."""
        result = cast_types(pre_cast_dataframe)
        assert result["exang"].dtype == "boolean"

    def test_exang_one_maps_to_true(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify exang '1' maps to True."""
        result = cast_types(pre_cast_dataframe)
        assert result.loc[1, "exang"]

    def test_exang_zero_maps_to_false(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify exang '0' maps to False."""
        result = cast_types(pre_cast_dataframe)
        assert not result.loc[0, "exang"]

    def test_disease_dtype_is_boolean(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify disease is cast to boolean (nullable)."""
        result = cast_types(pre_cast_dataframe)
        assert result["disease"].dtype == "boolean"

    def test_disease_zero_maps_to_false(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify disease '0' maps to False."""
        result = cast_types(pre_cast_dataframe)
        assert not result.loc[0, "disease"]

    def test_disease_one_maps_to_true(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify disease '1' maps to True."""
        result = cast_types(pre_cast_dataframe)
        assert result.loc[1, "disease"]

    def test_sex_dtype_is_category(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify sex is cast to category (unordered)."""
        result = cast_types(pre_cast_dataframe)
        assert isinstance(result["sex"].dtype, pd.CategoricalDtype)
        assert not result["sex"].dtype.ordered

    def test_chest_pain_dtype_is_category(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify chest_pain is cast to category (unordered)."""
        result = cast_types(pre_cast_dataframe)
        assert isinstance(result["chest_pain"].dtype, pd.CategoricalDtype)
        assert not result["chest_pain"].dtype.ordered

    def test_thal_dtype_is_category(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify thal is cast to category (unordered)."""
        result = cast_types(pre_cast_dataframe)
        assert isinstance(result["thal"].dtype, pd.CategoricalDtype)
        assert not result["thal"].dtype.ordered

    def test_rest_ecg_is_ordered_category(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify rest_ecg is cast to ordered category."""
        result = cast_types(pre_cast_dataframe)
        assert isinstance(result["rest_ecg"].dtype, pd.CategoricalDtype)
        assert result["rest_ecg"].dtype.ordered

    def test_rest_ecg_category_order(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify rest_ecg categories are in correct order."""
        result = cast_types(pre_cast_dataframe)
        expected_order = [
            "normal",
            "ST-T wave abnormality",
            "left ventricular hypertrophy",
        ]
        assert list(result["rest_ecg"].dtype.categories) == expected_order

    def test_slope_is_ordered_category(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify slope is cast to ordered category."""
        result = cast_types(pre_cast_dataframe)
        assert isinstance(result["slope"].dtype, pd.CategoricalDtype)
        assert result["slope"].dtype.ordered

    def test_slope_category_order(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify slope categories are in correct order."""
        result = cast_types(pre_cast_dataframe)
        expected_order = ["1", "2", "3"]
        assert list(result["slope"].dtype.categories) == expected_order

    def test_ca_is_ordered_category(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify ca is cast to ordered category."""
        result = cast_types(pre_cast_dataframe)
        assert isinstance(result["ca"].dtype, pd.CategoricalDtype)
        assert result["ca"].dtype.ordered

    def test_ca_category_order(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify ca categories are in correct order."""
        result = cast_types(pre_cast_dataframe)
        expected_order = ["0.0", "1.0", "2.0", "3.0"]
        assert list(result["ca"].dtype.categories) == expected_order

    def test_nan_preserved_in_int64(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify NaN values are preserved in Int64 columns."""
        result = cast_types(pre_cast_dataframe)
        assert pd.isna(result.loc[1, "age"])
        assert pd.isna(result.loc[2, "rest_bp"])

    def test_nan_preserved_in_boolean(self, pre_cast_dataframe: pd.DataFrame) -> None:
        """Verify NaN values are preserved in boolean columns."""
        result = cast_types(pre_cast_dataframe)
        assert pd.isna(result.loc[2, "exang"])
        assert pd.isna(result.loc[2, "disease"])
