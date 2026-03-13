"""Data loading and exploration functions for heart disease dataset."""

from pathlib import Path

import pandas as pd


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load heart disease dataset from CSV file.

    Args:
        file_path: Path object pointing to the CSV file.

    Returns:
        DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    return pd.read_csv(file_path)


def explore_data(df: pd.DataFrame) -> None:
    """
    Display dataset information: types, sample rows, and basic statistics.

    Args:
        df: DataFrame to explore.
    """
    print("\n" + "=" * 80)
    print("DATA TYPES AND MISSING VALUES")
    print("=" * 80)
    df.info()

    print("\n" + "=" * 80)
    print("SAMPLE (UP TO 10 RANDOM ROWS)")
    print("=" * 80)
    sample_size = min(10, len(df))
    if sample_size > 0:
        print(df.sample(sample_size))
    else:
        print("(empty dataset)")

    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    print(df.describe())


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save DataFrame to parquet format.

    Args:
        df: DataFrame to save.
        output_path: Path object where parquet file will be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
