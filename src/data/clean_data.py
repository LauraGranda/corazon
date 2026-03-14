"""Data cleaning and validation functions for heart disease dataset."""

import numpy as np
import pandas as pd

# Schema constants — business rules from datos_corazon_Info.txt
VALID_SEX: set[str] = {"Male", "Female"}
VALID_CHEST_PAIN: set[str] = {"typical", "asymptomatic", "nonanginal", "nontypical"}
VALID_REST_ECG: set[str] = {
    "normal",
    "left ventricular hypertrophy",
    "ST-T wave abnormality",
}
VALID_EXANG: set[str] = {"0", "1"}
VALID_SLOPE: set[str] = {"1", "2", "3"}
VALID_CA: set[str] = {"0.0", "1.0", "2.0", "3.0"}
VALID_THAL: set[str] = {"normal", "fixed", "reversable"}
VALID_DISEASE: set[str] = {"0", "1"}


def standardize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace irregular null representations with np.nan across all columns.

    Args:
        df: DataFrame with potentially irregular null representations.

    Returns:
        DataFrame with all nulls standardized to np.nan.
    """
    null_patterns: list[str] = ["?", "NA", "null", "NULL", "None", "none", ""]
    return df.replace(null_patterns, np.nan)


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from all object columns.

    Args:
        df: DataFrame with potentially whitespace-padded string values.

    Returns:
        DataFrame with whitespace stripped from all object-dtype columns.
    """
    result: pd.DataFrame = df.copy()
    str_cols: list[str] = result.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        result[col] = result[col].str.strip()
    return result


def invalidate_categorical(series: pd.Series, valid_values: set[str]) -> pd.Series:
    """
    Replace values not in valid_values with np.nan.

    Generic helper for validating categorical columns against an allowed set.

    Args:
        series: Series to validate.
        valid_values: Set of acceptable string values.

    Returns:
        Series with invalid values replaced by np.nan.
    """
    return series.where(series.isin(valid_values))


def invalidate_numeric(series: pd.Series) -> pd.Series:
    """
    Convert to numeric, coercing non-parseable values to NaN.

    Generic helper for validating numeric columns.

    Args:
        series: Series with values to convert to numeric.

    Returns:
        Series converted to float64, with non-numeric values as NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate all columns against schema constants.

    Applies categorical validation (using VALID_* sets) and numeric validation
    to the appropriate columns. Invalid values become NaN.

    Args:
        df: DataFrame with standardized nulls and cleaned strings.

    Returns:
        DataFrame with all invalid values replaced by NaN.
    """
    result: pd.DataFrame = df.copy()

    # Validate categorical columns
    result["sex"] = invalidate_categorical(result["sex"], VALID_SEX)
    result["chest_pain"] = invalidate_categorical(result["chest_pain"], VALID_CHEST_PAIN)
    result["rest_ecg"] = invalidate_categorical(result["rest_ecg"], VALID_REST_ECG)
    result["exang"] = invalidate_categorical(result["exang"], VALID_EXANG)
    result["slope"] = invalidate_categorical(result["slope"], VALID_SLOPE)
    result["ca"] = invalidate_categorical(result["ca"], VALID_CA)
    result["thal"] = invalidate_categorical(result["thal"], VALID_THAL)
    result["disease"] = invalidate_categorical(result["disease"], VALID_DISEASE)

    # Validate numeric columns
    result["age"] = invalidate_numeric(result["age"])
    result["rest_bp"] = invalidate_numeric(result["rest_bp"])
    result["chol"] = invalidate_numeric(result["chol"])
    result["max_hr"] = invalidate_numeric(result["max_hr"])
    result["old_peak"] = invalidate_numeric(result["old_peak"])

    return result


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all columns to their proper domain types.

    Converts object dtype columns to appropriate pandas types:
    - Nullable integers (Int64) for age, rest_bp, chol, max_hr
    - Float64 for old_peak
    - Boolean for fbs, exang, disease (with null-aware handling)
    - Unordered categories for sex, chest_pain, thal
    - Ordered categories for rest_ecg, slope, ca

    Args:
        df: DataFrame with validated values, numeric columns as float or int.

    Returns:
        DataFrame with all columns cast to proper types.
    """
    result: pd.DataFrame = df.copy()

    # Nullable integers
    result["age"] = result["age"].astype("Int64")
    result["rest_bp"] = result["rest_bp"].astype("Int64")
    result["chol"] = result["chol"].astype("Int64")
    result["max_hr"] = result["max_hr"].astype("Int64")

    # Float
    result["old_peak"] = result["old_peak"].astype("float64")

    # Boolean columns
    result["fbs"] = result["fbs"].astype("boolean")
    result["exang"] = result["exang"].map({"1": True, "0": False}).astype("boolean")
    result["disease"] = result["disease"].map({"1": True, "0": False}).astype("boolean")

    # Nominal categoricals (unordered)
    result["sex"] = result["sex"].astype("category")
    result["chest_pain"] = result["chest_pain"].astype("category")
    result["thal"] = result["thal"].astype("category")

    # Ordinal categoricals (ordered)
    rest_ecg_dtype = pd.CategoricalDtype(
        categories=["normal", "ST-T wave abnormality", "left ventricular hypertrophy"],
        ordered=True,
    )
    result["rest_ecg"] = result["rest_ecg"].astype(rest_ecg_dtype)

    slope_dtype = pd.CategoricalDtype(categories=["1", "2", "3"], ordered=True)
    result["slope"] = result["slope"].astype(slope_dtype)

    ca_dtype = pd.CategoricalDtype(categories=["0.0", "1.0", "2.0", "3.0"], ordered=True)
    result["ca"] = result["ca"].astype(ca_dtype)

    return result
