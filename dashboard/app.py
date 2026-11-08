from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures" / "FD001"
ARTIFACTS = ROOT / "artifacts" / "FD001"
EXPLANATIONS = ARTIFACTS / "explanations"
SPLITS = ROOT / "data" / "processed" / "FD001" / "splits"

st.set_page_config(
    page_title="DefenseAI | Engine Intelligence",
    layout="wide",
)


@st.cache_data
def read_csv(path, modified_time):
    return pd.read_csv(path)


@st.cache_data
def read_json(path, modified_time):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_file(path):
    return read_csv(str(path), path.stat().st_mtime_ns)


def json_file(path):
    return read_json(str(path), path.stat().st_mtime_ns)


def show_image(filename, caption):
    path = FIGURES / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chart not found: {filename}")


required = [
    REPORTS / "fd001_model_comparison.csv",
    REPORTS / "fd001_validation_error_details.csv",
    REPORTS / "fd001_errors_by_engine.csv",
    REPORTS / "fd001_errors_by_rul_band.csv",
    REPORTS / "fd001_error_summary.json",
    ARTIFACTS / "training_summary.json",
    SPLITS / "validation_split.csv",
]

missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]

if missing:
    st.error("Required project files are missing.")
    st.write(missing)
    st.stop()

comparison = csv_file(REPORTS / "fd001_model_comparison.csv")
predictions = csv_file(REPORTS / "fd001_validation_error_details.csv")
engine_errors = csv_file(REPORTS / "fd001_errors_by_engine.csv")
band_errors = csv_file(REPORTS / "fd001_errors_by_rul_band.csv")
error_summary = json_file(REPORTS / "fd001_error_summary.json")
training_summary = json_file(ARTIFACTS / "training_summary.json")

winner = training_summary["selected_model"]
prediction_column = f"{winner}_prediction"

st.sidebar.title("DefenseAI")
st.sidebar.caption("Equipment Intelligence Prototype")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Engine Explorer", "Error Analysis", "Explainability"],
)

st.sidebar.divider()
st.sidebar.write("Dataset: NASA C-MAPSS FD001")
st.sidebar.write(f"Selected model: {winner}")
st.sidebar.caption(
    "Historical validation data only. "
    "Official test data has not been evaluated."
)

if st.sidebar.button("Reload saved results"):
    st.cache_data.clear()
    st.rerun()

st.title("DefenseAI")
st.caption(
    "Aircraft-engine degradation analysis | "
    "Public simulated research data | Portfolio prototype"
)

st.warning(
    "Research demonstration only. Predictions are not certified "
    "maintenance guidance, failure probabilities, or readiness scores."
)


if page == "Overview":
    st.header("Validation Overview")

    overall = error_summary["overall"]

    a, b, c, d = st.columns(4)
    a.metric("Validation engines", predictions["engine_id"].nunique())
    b.metric("Validation records", f"{len(predictions):,}")
    c.metric("MAE", f"{overall['mae_cycles']:.2f} cycles")
    d.metric("RMSE", f"{overall['rmse_cycles']:.2f} cycles")

    st.subheader("Model Comparison")
    st.dataframe(comparison.round(2), hide_index=True, width="stretch")

    chart_data = comparison.melt(
        id_vars="model",
        value_vars=[
            "validation_mae_cycles",
            "validation_rmse_cycles",
        ],
        var_name="Metric",
        value_name="Error (cycles)",
    )

    fig = px.bar(
        chart_data,
        x="model",
        y="Error (cycles)",
        color="Metric",
        barmode="group",
        title="Validation Error by Model",
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "Scores cover all cycles from 20 held-out engines. "
        "Longer trajectories contribute more rows. Validation was "
        "also used for model selection and XGBoost early stopping."
    )

    st.download_button(
        "Download model comparison",
        comparison.to_csv(index=False),
        file_name="defenseai_model_comparison.csv",
        mime="text/csv",
    )


elif page == "Engine Explorer":
    st.header("Historical Engine Explorer")

    engine_ids = sorted(predictions["engine_id"].unique().tolist())
    engine_id = st.selectbox("Validation engine", engine_ids)

    engine = predictions.loc[
        predictions["engine_id"] == engine_id
    ].sort_values("cycle").reset_index(drop=True)

    cycles = engine["cycle"].astype(int).tolist()

    selected_cycle = st.select_slider(
        "Observed operating cycle",
        options=cycles,
        value=cycles[len(cycles) // 2],
    )

    row = engine.loc[engine["cycle"] == selected_cycle].iloc[0]

    a, b, c = st.columns(3)
    a.metric("Predicted RUL", f"{row[prediction_column]:.2f} cycles")
    b.metric("Actual RUL — retrospective", f"{row['rul']:.0f} cycles")
    c.metric(
        "Signed error",
        f"{row['error_cycles']:+.2f} cycles",
    )

    st.caption(
        "Positive error means remaining life was overestimated. "
        "Actual RUL is available here because this is retrospective "
        "evaluation; it would not be known during live operation."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=engine["cycle"],
        y=engine["rul"],
        name="Actual RUL",
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=engine["cycle"],
        y=engine[prediction_column],
        name="Predicted RUL",
        mode="lines",
    ))
    fig.add_vline(
        x=selected_cycle,
        line_dash="dash",
        line_color="gray",
    )
    fig.update_layout(
        title=f"Engine {engine_id}: Full Historical Trajectory",
        xaxis_title="Operating cycle",
        yaxis_title="Remaining useful life (cycles)",
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "This chart shows the complete saved trajectory, including "
        "cycles after the selected record. It is not a live replay."
    )

    validation = csv_file(SPLITS / "validation_split.csv")
    sensor_columns = [
        column for column in validation.columns
        if column.startswith("sensor_")
    ]

    sensor = st.selectbox("Sensor trend", sensor_columns)

    sensor_data = validation.loc[
        validation["engine_id"] == engine_id
    ].sort_values("cycle")

    sensor_fig = px.line(
        sensor_data,
        x="cycle",
        y=sensor,
        title=f"Engine {engine_id}: {sensor}",
    )
    st.plotly_chart(sensor_fig, width="stretch")

    with st.expander("Selected record and saved predictions"):
        st.dataframe(
            engine.loc[engine["cycle"] == selected_cycle],
            hide_index=True,
            width="stretch",
        )

    st.download_button(
        "Download this engine's predictions",
        engine.to_csv(index=False),
        file_name=f"engine_{engine_id}_validation_predictions.csv",
        mime="text/csv",
    )


elif page == "Error Analysis":
    st.header("Validation Error Analysis")

    a, b, c = st.columns(3)
    a.metric(
        "Mean signed error",
        f"{error_summary['overall']['mean_error_cycles']:+.2f} cycles",
    )
    b.metric(
        "Records with overestimated RUL",
        f"{error_summary['overall']['overestimate_percent']:.2f}%",
    )
    c.metric(
        "Equal-engine-weighted MAE",
        f"{error_summary['equal_engine_weighted_mae_cycles']:.2f} cycles",
    )

    st.subheader("Error by Actual Remaining-Life Range")
    st.dataframe(band_errors.round(2), hide_index=True, width="stretch")

    band_fig = px.bar(
        band_errors,
        x="actual_rul_band",
        y="mae_cycles",
        title="MAE by Actual RUL Range",
        labels={
            "actual_rul_band": "Actual RUL range (cycles)",
            "mae_cycles": "MAE (cycles)",
        },
    )
    st.plotly_chart(band_fig, width="stretch")

    st.caption(
        "Ranges are based on known evaluation labels. "
        "They are not deployable maintenance thresholds."
    )

    st.subheader("Per-Engine Errors")
    st.dataframe(
        engine_errors.round(2),
        hide_index=True,
        width="stretch",
    )

    histogram = px.histogram(
        predictions,
        x="error_cycles",
        nbins=50,
        title="Signed Prediction Error Distribution",
        labels={"error_cycles": "Predicted RUL - actual RUL (cycles)"},
    )
    histogram.add_vline(x=0, line_dash="dash")
    st.plotly_chart(histogram, width="stretch")


elif page == "Explainability":
    st.header("SHAP Model Explanations")

    st.info(
        "SHAP explains the raw regression output before its "
        "non-negative floor. Positive contributions increase predicted "
        "remaining life; negative contributions decrease it. "
        "These are model attributions, not physical fault diagnoses."
    )

    global_path = REPORTS / "fd001_shap_global_importance.csv"

    if global_path.exists():
        st.subheader("Global Feature Contributions")
        importance = csv_file(global_path)
        st.dataframe(
            importance.round(3),
            hide_index=True,
            width="stretch",
        )

    show_image(
        "12_shap_global_importance.png",
        "Mean absolute contributions across 500 sampled validation records.",
    )
    show_image(
        "13_shap_beeswarm.png",
        "Feature-value patterns and their contributions to predicted RUL.",
    )

    st.divider()
    st.subheader("Saved Local Diagnostic Example")

    local_path = EXPLANATIONS / "local_explanation.json"

    if local_path.exists():
        local = json_file(local_path)

        st.write(
            f"Engine {local['engine_id']} | "
            f"Operating cycle {local['cycle']}"
        )

        a, b, c = st.columns(3)
        a.metric(
            "Actual RUL",
            f"{local['actual_rul_cycles']:.2f} cycles",
        )
        b.metric(
            "Predicted RUL",
            f"{local['displayed_predicted_rul_cycles']:.2f} cycles",
        )
        c.metric(
            "Reference prediction",
            f"{local['base_rul_cycles']:.2f} cycles",
        )

        st.caption(
            "This is a fixed, deliberately selected diagnostic record. "
            "It does not change with the Engine Explorer selection."
        )
        st.write("Selection rule:", local["selection_rule"])

        show_image(
            "14_shap_local_waterfall.png",
            "Local contributions from reference prediction to raw model output.",
        )

        st.dataframe(
            pd.DataFrame(local["top_contributions"]).round(3),
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download local explanation",
            json.dumps(local, indent=2),
            file_name="defenseai_local_explanation.json",
            mime="application/json",
        )
    else:
        st.info("Run explain_model.py to create the local explanation.")

st.divider()
st.caption(
    "Current scope: predictive ML, validation diagnostics, and saved "
    "SHAP explanations. RAG, fine-tuning, agents, and financial "
    "simulation are not implemented in this version."
)