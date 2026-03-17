# HOW TO RUN THE APP: streamlit run notebooks/7-deploy/app.py

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

sys.path.append(str(Path.cwd().resolve()))

from src.inference.predict import load_model as backend_load_model
from src.inference.predict import make_predictions

REQUIRED_COLUMNS: list[str] = [
    "age",
    "max_hr",
    "old_peak",
    "chest_pain",
    "sex",
    "thal",
    "slope",
    "ca",
    "exang",
]


def get_user_data() -> pd.DataFrame:
    """Gather clinical features from user via Streamlit widgets."""
    st.subheader("Enter Patient Clinical Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=100, value=50, step=1)
        max_hr = st.slider(
            "Max Heart Rate (bpm)",
            min_value=60,
            max_value=220,
            value=150,
            step=1,
        )
        old_peak = st.number_input(
            "ST Depression (old_peak)",
            min_value=0.0,
            max_value=6.2,
            value=1.0,
            step=0.1,
        )

    with col2:
        chest_pain_options = {
            "Typical Angina": "typical",
            "Atypical Angina": "nontypical",
            "Non-anginal Pain": "nonanginal",
            "Asymptomatic": "asymptomatic",
        }
        chest_pain_label = st.selectbox("Chest Pain Type", list(chest_pain_options.keys()))
        chest_pain = chest_pain_options[chest_pain_label]

        sex_label = st.radio("Sex", ["Female", "Male"])
        sex = sex_label

        thal_options = {
            "Normal": 1,
            "Fixed Defect": 2,
            "Reversible Defect": 3,
        }
        thal_label = st.selectbox("Thalassemia Type (thal)", list(thal_options.keys()))
        thal = thal_options[thal_label]

    with col3:
        slope_options = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2,
        }
        slope_label = st.selectbox("ST Slope", list(slope_options.keys()))
        slope = slope_options[slope_label]

        ca = st.selectbox(
            "Major Vessels Colored (ca)",
            [0, 1, 2, 3],
            index=0,
        )

        exang_label = st.radio("Exercise Induced Angina (exang)", ["No", "Yes"])
        exang = 0 if exang_label == "No" else 1

    user_data: dict = {
        "age": [age],
        "max_hr": [max_hr],
        "old_peak": [old_peak],
        "chest_pain": [chest_pain],
        "sex": [sex],
        "thal": [thal],
        "slope": [slope],
        "ca": [ca],
        "exang": [exang],
    }

    df = pd.DataFrame(user_data)
    return df


def preprocess_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess batch CSV data for prediction."""
    df_copy = df.copy()

    # Map chest_pain and sex to lowercase string values (as expected by model)
    chest_pain_mapping = {
        "Typical Angina": "typical",
        "Atypical Angina": "nontypical",
        "Non-anginal Pain": "nonanginal",
        "Asymptomatic": "asymptomatic",
        "typical": "typical",
        "nontypical": "nontypical",
        "nonanginal": "nonanginal",
        "asymptomatic": "asymptomatic",
    }

    sex_mapping = {
        "Female": "Female",
        "Male": "Male",
        "F": "Female",
        "M": "Male",
        "0": "Female",
        "1": "Male",
    }

    if "chest_pain" in df_copy.columns and df_copy["chest_pain"].dtype == "object":
        df_copy["chest_pain"] = df_copy["chest_pain"].map(
            lambda x: chest_pain_mapping.get(str(x).strip(), str(x).lower())
        )

    if "sex" in df_copy.columns and df_copy["sex"].dtype == "object":
        df_copy["sex"] = df_copy["sex"].map(lambda x: sex_mapping.get(str(x).strip(), str(x)))

    # Cast numeric columns
    numeric_cols = ["age", "max_hr", "old_peak", "ca", "thal", "slope", "exang"]
    for col in numeric_cols:
        if col in df_copy.columns and col not in ["chest_pain", "sex"]:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    return df_copy[REQUIRED_COLUMNS]


@st.cache_resource
def load_model_cached(model_dir: Path, model_name: str) -> Pipeline:
    """Load the trained LogisticRegression pipeline using backend inference."""
    with st.spinner("Loading model..."):
        return backend_load_model(model_dir=model_dir, model_name=model_name)


def individual_prediction_tab(model: Pipeline) -> None:
    """Predict disease risk for a single patient."""
    st.header("Individual Prediction")

    df_user_data = get_user_data()

    if st.button("Predict Disease Risk"):
        predictions = make_predictions(model, df_user_data)
        state = predictions[0]

        st.title("Diagnosis Prediction")

        if state == 1:
            st.error(
                "⚠️ **High Risk for Heart Disease**\n\n"
                "This patient profile indicates a significant risk of heart disease. "
                "Further clinical evaluation and cardiac testing are recommended."
            )
        else:
            st.success(
                "✅ **Healthy Patient Profile**\n\n"
                "Based on the provided clinical features, this patient shows a healthy profile. "
                "Continue with regular preventive care."
            )


def batch_prediction_tab(model: Pipeline) -> None:
    """Predict disease risk for multiple patients via CSV upload."""
    st.header("Batch Prediction")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)

        # Check if required columns exist
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_batch.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info(
                f"Please ensure your CSV contains all these columns: {', '.join(REQUIRED_COLUMNS)}"
            )
        else:
            if st.button("Predict Disease Risk for All Patients"):
                df_processed = preprocess_batch_data(df_batch)
                predictions = make_predictions(model, df_processed)

                df_results = df_batch.copy()
                df_results["Predicted_Disease"] = predictions
                df_results["Diagnosis"] = df_results["Predicted_Disease"].map(
                    {0: "Healthy", 1: "At Risk"}
                )

                st.subheader("Prediction Results")
                st.dataframe(df_results, use_container_width=True)

                # Calculate and display overall risk rate
                risk_rate = (predictions == 1).sum() / len(predictions) * 100
                st.metric("Overall Risk Rate", f"{risk_rate:.1f}%")

                # Download button
                csv_data = df_results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_data,
                    file_name="disease_predictions.csv",
                    mime="text/csv",
                )

            # Show sample format
            st.subheader("📋 Expected CSV Format")
            sample_data = {col: [0 if col != "age" else 50] for col in REQUIRED_COLUMNS}
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)


def main() -> None:
    """Main application logic."""
    st.set_page_config(
        page_title="Heart Disease Clinical Assistant",
        page_icon="🫀",
        layout="wide",
    )

    st.header("Heart Disease Clinical Assistant 🫀")

    models_path = Path.cwd().resolve() / "models"
    model = load_model_cached(
        model_dir=models_path,
        model_name="simple_logistic_regression.joblib",
    )

    tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])

    with tab1:
        individual_prediction_tab(model)

    with tab2:
        batch_prediction_tab(model)


if __name__ == "__main__":
    main()
