from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "processed" / "FD001" / "splits"
MODEL_DIR = ROOT / "artifacts" / "FD001"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "FD001"
EXPLANATION_DIR = MODEL_DIR / "explanations"

SEED = 42


def save_chart(filename):
    plt.gcf().savefig(
        FIGURE_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()
    print(f"[CHART] {filename}")


def main():
    for directory in [FIGURE_DIR, EXPLANATION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_DIR / "best_model.joblib")
    model = bundle["model"]
    features = bundle["feature_columns"]

    if not isinstance(model, xgb.XGBRegressor):
        raise TypeError(
            "This script expects the selected model to be XGBoost."
        )

    train = pd.read_csv(SPLIT_DIR / "train_split.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation_split.csv")

    if set(train["engine_id"]) & set(validation["engine_id"]):
        raise ValueError("Training and validation engines overlap.")

    # Background records come from the training partition only.
    background = train[features].sample(
        n=min(100, len(train)),
        random_state=SEED,
    )

    # Global plots summarize this random sample, not all validation rows.
    sample = validation.sample(
        n=min(500, len(validation)),
        random_state=SEED,
    ).reset_index(drop=True)

    # Deliberate diagnostic example: largest positive prediction error
    # among validation records whose actual RUL is at most 30 cycles.
    near_failure = validation.loc[validation["rul"] <= 30].copy()

    if near_failure.empty:
        raise ValueError("No validation records with RUL <= 30.")

    near_failure["predicted_rul"] = np.maximum(
        model.predict(near_failure[features]), 0
    )
    near_failure["error_cycles"] = (
        near_failure["predicted_rul"] - near_failure["rul"]
    )

    local = near_failure.loc[
        [near_failure["error_cycles"].idxmax()]
    ].reset_index(drop=True)

    X_sample = sample[features]
    X_local = local[features]

    # Restrict the booster to the trees used by early-stopped predict().
    booster = model.get_booster()

    try:
        rounds = int(model.best_iteration) + 1
    except AttributeError:
        rounds = booster.num_boosted_rounds()

    explanation_booster = booster[:rounds]

    X_check = pd.concat(
        [X_sample, X_local],
        ignore_index=True,
    )

    wrapper_predictions = model.predict(X_check)
    booster_predictions = explanation_booster.predict(
        xgb.DMatrix(X_check)
    )

    np.testing.assert_allclose(
        booster_predictions,
        wrapper_predictions,
        rtol=1e-5,
        atol=1e-3,
        err_msg="Explanation booster differs from deployed model.",
    )

    print("=" * 65)
    print("DefenseAI: SHAP explanations")
    print("=" * 65)
    print(f"Background training records: {len(background)}")
    print(f"Global validation sample:    {len(sample)}")
    print(f"Boosting rounds explained:   {rounds}")
    print("\nComputing SHAP values...", flush=True)

    explainer = shap.TreeExplainer(
        explanation_booster,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )

    global_explanation = explainer(X_sample)
    local_explanation = explainer(X_local)

    reconstructed = (
        global_explanation.base_values
        + global_explanation.values.sum(axis=1)
    )

    np.testing.assert_allclose(
        reconstructed,
        model.predict(X_sample),
        rtol=1e-4,
        atol=1e-2,
        err_msg="SHAP contributions do not reconstruct predictions.",
    )

    importance = pd.DataFrame({
        "feature": features,
        "mean_absolute_shap_cycles": np.abs(
            global_explanation.values
        ).mean(axis=0),
    }).sort_values(
        "mean_absolute_shap_cycles",
        ascending=False,
    )

    importance.to_csv(
        REPORT_DIR / "fd001_shap_global_importance.csv",
        index=False,
    )

    sample_values = pd.DataFrame(
        global_explanation.values,
        columns=[f"{feature}_shap_cycles" for feature in features],
    )

    sample_export = pd.concat(
        [
            sample[["engine_id", "cycle", "rul"]],
            pd.Series(
                global_explanation.base_values,
                name="base_rul_cycles",
            ),
            pd.Series(
                model.predict(X_sample),
                name="raw_predicted_rul",
            ),
            sample_values,
        ],
        axis=1,
    )

    sample_export.to_csv(
        EXPLANATION_DIR / "validation_sample_shap.csv",
        index=False,
    )

    background.to_csv(
        EXPLANATION_DIR / "background_training_rows.csv",
        index=False,
    )

    local_values = local_explanation.values[0]
    base_value = float(local_explanation.base_values[0])
    raw_prediction = float(model.predict(X_local)[0])

    np.testing.assert_allclose(
        base_value + local_values.sum(),
        raw_prediction,
        rtol=1e-4,
        atol=1e-2,
    )

    contributions = pd.DataFrame({
        "feature": features,
        "sensor_or_setting_value": X_local.iloc[0].to_numpy(),
        "shap_cycles": local_values,
    })

    contributions["absolute_shap_cycles"] = (
        contributions["shap_cycles"].abs()
    )
    contributions = contributions.sort_values(
        "absolute_shap_cycles",
        ascending=False,
    )

    contributions.to_csv(
        EXPLANATION_DIR / "local_feature_contributions.csv",
        index=False,
    )

    local_report = {
        "engine_id": int(local.iloc[0]["engine_id"]),
        "cycle": int(local.iloc[0]["cycle"]),
        "actual_rul_cycles": float(local.iloc[0]["rul"]),
        "raw_predicted_rul_cycles": raw_prediction,
        "displayed_predicted_rul_cycles": max(raw_prediction, 0),
        "base_rul_cycles": base_value,
        "sum_shap_cycles": float(local_values.sum()),
        "selection_rule": (
            "Largest positive prediction error among validation "
            "records with actual RUL <= 30."
        ),
        "explained_output": (
            "Raw regression prediction before non-negative clipping."
        ),
        "background": "100 randomly sampled training rows",
        "feature_perturbation": "interventional",
        "top_contributions": contributions.head(10).to_dict(
            orient="records"
        ),
        "limitations": [
            "Explains the model, not physical fault causality.",
            "Global importance uses a 500-row validation sample.",
            "Correlated features and reference data affect attributions.",
            "The local example is deliberately selected, not representative.",
            "Official test data was not used.",
        ],
    }

    (EXPLANATION_DIR / "local_explanation.json").write_text(
        json.dumps(local_report, indent=2),
        encoding="utf-8",
    )

    shap.plots.bar(
        global_explanation,
        max_display=15,
        show=False,
    )
    save_chart("12_shap_global_importance.png")

    shap.plots.beeswarm(
        global_explanation,
        max_display=15,
        show=False,
    )
    save_chart("13_shap_beeswarm.png")

    shap.plots.waterfall(
        local_explanation[0],
        max_display=12,
        show=False,
    )
    save_chart("14_shap_local_waterfall.png")

    print("\nTOP 10 FEATURES")
    print(
        importance.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nLOCAL VALIDATION EXAMPLE")
    print(f"Engine:        {local_report['engine_id']}")
    print(f"Cycle:         {local_report['cycle']}")
    print(f"Actual RUL:    {local_report['actual_rul_cycles']:.2f}")
    print(f"Predicted RUL: {max(raw_prediction, 0):.2f}")
    print(f"SHAP baseline: {base_value:.2f}")

    print("\nTop local contributions:")
    print(
        contributions.head(8).to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nSUCCESS: SHAP explanations created.")
    print("Model prediction and SHAP reconstruction checks passed.")
    print("No retraining. Official test data remains untouched.")


if __name__ == "__main__":
    main()