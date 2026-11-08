from pathlib import Path
from io import StringIO
import csv

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "FD001" / "best_model.joblib"
SPLIT_DIR = ROOT / "data" / "processed" / "FD001" / "splits"

st.set_page_config(
    page_title="DefenseAI | Sensor Inference",
    layout="wide",
)


@st.cache_resource
def load_bundle(path, modified_time):
    # Load only the trusted model created locally by this project.
    return joblib.load(path)


@st.cache_data
def load_csv(path, modified_time):
    return pd.read_csv(path)


def local_csv(path):
    return load_csv(str(path), path.stat().st_mtime_ns)


st.title("DefenseAI — Sensor Inference")
st.caption("Generate RUL predictions from uploaded sensor records.")

st.warning(
    "Research prototype for C-MAPSS FD001-compatible simulated data. "
    "Not validated for real aircraft, other engine types, or "
    "operational maintenance decisions."
)

required_paths = [
    MODEL_PATH,
    SPLIT_DIR / "train_split.csv",
    SPLIT_DIR / "validation_split.csv",
]

missing_paths = [
    str(path.relative_to(ROOT))
    for path in required_paths
    if not path.exists()
]

if missing_paths:
    st.error("Required local project files are missing.")
    st.write(missing_paths)
    st.stop()

bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
model = bundle["model"]
features = bundle["feature_columns"]

training = local_csv(SPLIT_DIR / "train_split.csv")
validation = local_csv(SPLIT_DIR / "validation_split.csv")

minimums = training[features].min()
maximums = training[features].max()

st.subheader("1. Prepare your input")

with st.expander("Required input columns"):
    st.write(features)
    st.caption(
        "Feature order in the uploaded CSV can differ. "
        "The application selects and orders features using "
        "the saved model configuration."
    )

template = pd.DataFrame(columns=features)

# Known validation rows are reused only for an inference smoke test.
# This is not a new evaluation dataset.
demo = validation[
    ["engine_id", "cycle"] + features
].sample(
    n=min(10, len(validation)),
    random_state=42,
).sort_values(
    ["engine_id", "cycle"]
).reset_index(drop=True)

a, b = st.columns(2)

a.download_button(
    "Download empty CSV template",
    template.to_csv(index=False),
    file_name="defenseai_sensor_template.csv",
    mime="text/csv",
)

b.download_button(
    "Download 10-row demo CSV",
    demo.to_csv(index=False),
    file_name="defenseai_demo_sensor_records.csv",
    mime="text/csv",
)

st.caption(
    "The demo contains previously used validation records with "
    "RUL labels removed. It tests the upload and prediction workflow; "
    "it does not demonstrate performance on new data."
)

st.subheader("2. Choose records")

mode = st.radio(
    "Input source",
    ["Built-in validation demo", "Upload CSV"],
    horizontal=True,
)

if mode == "Built-in validation demo":
    data = demo.copy()
else:
    uploaded = st.file_uploader(
        "Upload a sensor CSV",
        type=["csv"],
    )

    if uploaded is None:
        st.info("Upload a CSV to continue.")
        st.stop()

    if uploaded.size > 10 * 1024 * 1024:
        st.error("Please upload a file smaller than 10 MB.")
        st.stop()

    try:
        text = uploaded.getvalue().decode("utf-8-sig")
        headers = next(csv.reader(StringIO(text)))

        if len(headers) != len(set(headers)):
            raise ValueError("Duplicate column names are not allowed.")

        data = pd.read_csv(StringIO(text))
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

if data.empty:
    st.error("The input has no data rows.")
    st.stop()

if len(data) > 10000:
    st.error("This prototype accepts up to 10,000 rows per file.")
    st.stop()

missing_features = [
    feature for feature in features if feature not in data.columns
]

if missing_features:
    st.error("Required features are missing.")
    st.write(missing_features)
    st.stop()

extra_columns = [
    column for column in data.columns
    if column not in features
]

if extra_columns:
    st.info(
        "Columns excluded from model inputs: "
        + ", ".join(extra_columns)
    )

X = data[features].apply(pd.to_numeric, errors="coerce")

valid_values = np.isfinite(X.to_numpy(dtype=float))
invalid_rows = ~valid_values.all(axis=1)

if invalid_rows.any():
    st.error(
        "Required features contain blank, nonnumeric, or infinite "
        "values. Correct these rows before predicting."
    )
    bad_records = data.loc[invalid_rows].copy()
    bad_records.insert(
        0,
        "input_row_number",
        np.flatnonzero(invalid_rows) + 1,
    )
    st.dataframe(bad_records.head(50), width="stretch")
    st.stop()

outside_range = X.lt(minimums) | X.gt(maximums)
outside_counts = outside_range.sum(axis=1)

st.success(
    f"Input validation passed: {len(X):,} records, "
    f"{len(features)} model features."
)

st.dataframe(data.head(20), hide_index=True, width="stretch")

if outside_counts.gt(0).any():
    st.warning(
        f"{int(outside_counts.gt(0).sum())} records contain feature "
        "values outside their observed training ranges. "
        "These predictions require additional scrutiny."
    )

st.caption(
    "The range check compares individual features with training "
    "minimums and maximums. Passing it does not establish that "
    "a record is in-distribution or that its prediction is reliable."
)

st.subheader("3. Generate predictions")

if st.button("Predict remaining useful life", type="primary"):
    try:
        raw_predictions = np.asarray(
            model.predict(X),
            dtype=float,
        )

        if not np.isfinite(raw_predictions).all():
            raise ValueError("The model returned non-finite predictions.")

        results = pd.DataFrame({
            "input_row_number": np.arange(1, len(X) + 1),
        })

        # Preserve optional identifiers as metadata, never model inputs.
        for column in ["engine_id", "cycle"]:
            if column in data.columns:
                results[column] = data[column].to_numpy()

        results["raw_predicted_rul_cycles"] = raw_predictions
        results["predicted_rul_cycles"] = np.maximum(
            raw_predictions,
            bundle.get("prediction_floor", 0),
        )
        results["outside_training_range_count"] = (
            outside_counts.to_numpy()
        )
        results["outside_training_range_features"] = [
            ", ".join(
                feature
                for feature in features
                if outside_range.iloc[index][feature]
            )
            for index in range(len(X))
        ]

        st.success("Predictions generated using the saved model.")

        a, b = st.columns(2)
        a.metric("Records predicted", f"{len(results):,}")
        b.metric(
            "Records with range warnings",
            int(outside_counts.gt(0).sum()),
        )

        st.dataframe(
            results.round(3),
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download predictions",
            results.to_csv(index=False),
            file_name="defenseai_inference_results.csv",
            mime="text/csv",
            on_click="ignore",
        )

        st.caption(
            "Outputs are operating-cycle estimates, not hours, "
            "failure probabilities, or confidence intervals. "
            "No actual RUL labels were used for prediction."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "This application loads the existing saved model. "
    "It does not retrain or access the official test dataset."
)