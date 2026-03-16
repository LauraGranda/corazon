"""Production training pipeline for heart disease LogisticRegression model."""

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from pipelines.feature_pipeline.build_features import create_preprocessor

# Load data
DATA_PATH: Path = Path("/home/lauragranda01/corazon/data/03_primary/corazon_explored.parquet")
dataset: pd.DataFrame = pd.read_parquet(DATA_PATH)

# Data preparation & Data preprocessing
target: str = "disease"
dataset = dataset.drop_duplicates()

# Feature selection
selected_features: list[str] = [
    "age",
    "max_hr",
    "old_peak",
    "chest_pain",
    "sex",
    "thal",
    "slope",
    "ca",
    "exang",
    target,
]
dataset = dataset[selected_features].copy()

# Convert data types to be compatible with sklearn pipeline
# Numeric columns to float (can handle NaN)
dataset[["age", "max_hr", "old_peak"]] = dataset[["age", "max_hr", "old_peak"]].astype(float)
# Categorical columns to string (including exang which is boolean)
dataset[["chest_pain", "sex", "thal", "slope", "ca", "exang"]] = dataset[
    ["chest_pain", "sex", "thal", "slope", "ca", "exang"]
].astype(str)
# Drop rows with NaN in target
dataset = dataset[dataset[target].notna()]
# Convert target to int
dataset[target] = dataset[target].astype(int)

# Train / Test split
X_features: pd.DataFrame = dataset.drop(target, axis="columns")
Y_target: pd.Series = dataset[target]

x_train, x_test, y_train, y_test = train_test_split(
    X_features, Y_target, test_size=0.2, random_state=42, stratify=Y_target
)

# Feature Engineering
preprocessor = create_preprocessor()
data_model_pipeline: Pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(solver="liblinear", random_state=42)),
    ]
)

# Hyperparameter tuning
score: str = "recall"
hyperparameters: dict = {
    "model__C": [0.1, 0.5, 1.0, 5.0],
    "model__penalty": ["l1", "l2"],
}
grid_search = GridSearchCV(data_model_pipeline, hyperparameters, cv=5, scoring=score, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_data_model_pipeline: Pipeline = grid_search.best_estimator_

# Evaluation
y_pred = best_data_model_pipeline.predict(x_test)
metric_result: float = recall_score(y_test, y_pred)
print(f"Recall on test set: {metric_result:.4f}")

# Model Validation
BASELINE_SCORE: float = 0.70
if metric_result > BASELINE_SCORE:
    print("Model validation passed")
else:
    print(f"Model validation failed: recall {metric_result:.4f} <= baseline {BASELINE_SCORE}")
    raise ValueError("Model did not meet baseline recall threshold")

# Save model
DATA_MODEL: Path = Path.cwd().resolve() / "models"
dump(best_data_model_pipeline, DATA_MODEL / "simple_logistic_regression.joblib", protocol=5)
print(f"Model saved to {DATA_MODEL / 'simple_logistic_regression.joblib'}")
