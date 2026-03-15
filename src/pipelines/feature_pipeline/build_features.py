"""Feature engineering pipeline module for heart disease classification."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Feature group constants
NUMERIC_FEATURES: list[str] = ["age", "max_hr", "old_peak"]
CATEGORICAL_FEATURES: list[str] = ["chest_pain", "sex"]
ORDINAL_FEATURES: list[str] = ["thal", "slope", "ca", "exang"]


def _build_numeric_pipeline() -> Pipeline:
    """Build the preprocessing pipeline for numerical features.

    Returns
    -------
    Pipeline
        Pipeline with median imputation for numerical features.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )


def _build_categorical_pipeline() -> Pipeline:
    """Build the preprocessing pipeline for nominal categorical features.

    Returns
    -------
    Pipeline
        Pipeline with most-frequent imputation and one-hot encoding.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def _build_ordinal_pipeline() -> Pipeline:
    """Build the preprocessing pipeline for ordinal categorical features.

    Returns
    -------
    Pipeline
        Pipeline with most-frequent imputation and ordinal encoding.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def create_preprocessor() -> ColumnTransformer:
    """Create and return a ColumnTransformer preprocessor for the heart disease dataset.

    The preprocessor combines three sub-pipelines for numerical, categorical,
    and ordinal features. It is returned unfitted and ready to be fit on
    training data.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor with three transformers:
        - "num": numerical features pipeline
        - "cat": categorical features pipeline
        - "ord": ordinal features pipeline
    """
    return ColumnTransformer(
        transformers=[
            ("num", _build_numeric_pipeline(), NUMERIC_FEATURES),
            ("cat", _build_categorical_pipeline(), CATEGORICAL_FEATURES),
            ("ord", _build_ordinal_pipeline(), ORDINAL_FEATURES),
        ]
    )
