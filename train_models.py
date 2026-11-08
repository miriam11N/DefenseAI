from pathlib import Path
from datetime import datetime, timezone
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent

SPLIT_DIR = ROOT / "data" / "processed" / "FD001" / "splits"
MODEL_DIR = ROOT / "artifacts" / "FD001"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "FD001"

SEED = 42


def calculate_metrics(actual, predicted):
    return {
        "mae_cycles": float(mean_absolute_error(actual, predicted)),
        "rmse_cycles": float(root_mean_squared_error(actual, predicted)),
    }


def main():
    for directory in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    config = json.loads(
        (SPLIT_DIR / "split_config.json").read_text(encoding="utf-8")
    )

    train = pd.read_csv(SPLIT_DIR / "train_split.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation_split.csv")

    features = config["feature_columns"]
    target = config["target_column"]

    if not features:
        raise ValueError("The feature list is empty.")

    forbidden = {"engine_id", "cycle", target}

    if forbidden.intersection(features):
        raise ValueError("Identifiers or target detected in model features.")

    train_ids = set(train["engine_id"])
    validation_ids = set(validation["engine_id"])

    if train_ids.intersection(validation_ids):
        raise ValueError("Training and validation engines overlap.")

    if train_ids != set(config["train_engine_ids"]):
        raise ValueError("Training engine IDs do not match the configuration.")

    if validation_ids != set(config["validation_engine_ids"]):
        raise ValueError("Validation engine IDs do not match the configuration.")

    X_train = train[features]
    y_train = train[target]
    X_validation = validation[features]
    y_validation = validation[target]

    for name, frame in [
        ("training", train[features + [target]]),
        ("validation", validation[features + [target]]),
    ]:
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid numeric values in {name} data.")

        if (frame[target] < 0).any():
            raise ValueError(f"Negative RUL in {name} data.")

    print("=" * 65)
    print("DefenseAI: First RUL model comparison")
    print("=" * 65)
    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Model features:  {len(features)}")
    print("Target: uncapped RUL in operating cycles")
    print("Official test data: not used")

    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=75,
            random_state=SEED,
            n_jobs=4,
        ),
    }

    results = []
    validation_predictions = validation[
        ["engine_id", "cycle", target]
    ].copy()

    for name, model in models.items():
        print(f"\n[TRAINING] {name}", flush=True)
        start = time.perf_counter()

        if name == "xgboost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_validation, y_validation)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        elapsed = time.perf_counter() - start

        # Apply the same non-negative output rule to every model.
        train_predictions = np.maximum(model.predict(X_train), 0)
        predictions = np.maximum(model.predict(X_validation), 0)

        train_metrics = calculate_metrics(y_train, train_predictions)
        validation_metrics = calculate_metrics(y_validation, predictions)

        row = {
            "model": name,
            "train_mae_cycles": train_metrics["mae_cycles"],
            "train_rmse_cycles": train_metrics["rmse_cycles"],
            "validation_mae_cycles": validation_metrics["mae_cycles"],
            "validation_rmse_cycles": validation_metrics["rmse_cycles"],
            "training_seconds": elapsed,
        }

        results.append(row)
        validation_predictions[f"{name}_prediction"] = predictions

        joblib.dump(
            {
                "model": model,
                "feature_columns": features,
                "target_column": target,
                "target_units": "operating cycles",
                "prediction_floor": 0,
                "target_cap": None,
            },
            MODEL_DIR / f"{name}.joblib",
        )

        print(
            f"[DONE] {name} | "
            f"Validation MAE: {validation_metrics['mae_cycles']:.2f} | "
            f"Validation RMSE: {validation_metrics['rmse_cycles']:.2f} | "
            f"Time: {elapsed:.1f}s",
            flush=True,
        )

        if name == "xgboost":
            model.save_model(MODEL_DIR / "xgboost.ubj")
            print(f"Best boosting round: {model.best_iteration + 1}")

            history = model.evals_result()["validation_0"]["rmse"]

            plt.figure(figsize=(9, 5))
            plt.plot(
                np.arange(1, len(history) + 1),
                history,
                color="steelblue",
            )
            plt.axvline(
                model.best_iteration + 1,
                color="darkorange",
                linestyle="--",
                label="Best round",
            )
            plt.xlabel("Boosting round")
            plt.ylabel("Validation RMSE (cycles)")
            plt.title("XGBoost Validation Learning Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                FIGURE_DIR / "05_xgboost_learning_curve.png",
                dpi=160,
            )
            plt.close()

    metrics = (
        pd.DataFrame(results)
        .sort_values("validation_rmse_cycles")
        .reset_index(drop=True)
    )

    metrics.to_csv(REPORT_DIR / "fd001_model_comparison.csv", index=False)

    validation_predictions.to_csv(
        REPORT_DIR / "fd001_validation_predictions.csv",
        index=False,
    )

    winner = str(metrics.iloc[0]["model"])
    winning_predictions = validation_predictions[
        f"{winner}_prediction"
    ].to_numpy()

    joblib.dump(
        joblib.load(MODEL_DIR / f"{winner}.joblib"),
        MODEL_DIR / "best_model.joblib",
    )

    run_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": winner,
        "selection_metric": "validation_rmse_cycles",
        "evaluation_population": "All rows of 20 held-out engines",
        "feature_columns": features,
        "prediction_floor": 0,
        "target_cap": None,
        "xgboost_best_round": models["xgboost"].best_iteration + 1,
        "official_test_used": False,
        "validation_used_for_xgboost_early_stopping": True,
        "results": metrics.to_dict(orient="records"),
        "limitations": [
            "One engine-level validation split only.",
            "Validation was used for model selection and early stopping.",
            "All-cycle metrics give longer engines more row weight.",
            "Validation metrics are not official test-set performance.",
            "Predicted RUL is not a calibrated failure probability.",
        ],
    }

    (MODEL_DIR / "training_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )

    positions = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        positions - width / 2,
        metrics["validation_mae_cycles"],
        width,
        label="MAE",
    )
    plt.bar(
        positions + width / 2,
        metrics["validation_rmse_cycles"],
        width,
        label="RMSE",
    )
    plt.xticks(positions, metrics["model"])
    plt.ylabel("Error (operating cycles)")
    plt.title("FD001: Validation Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "06_model_comparison.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(
        y_validation,
        winning_predictions,
        s=8,
        alpha=0.2,
    )
    upper = max(float(y_validation.max()), float(winning_predictions.max()))
    plt.plot([0, upper], [0, upper], "r--", label="Perfect prediction")
    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title(f"Validation Predictions: {winner}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "07_actual_vs_predicted.png",
        dpi=160,
    )
    plt.close()

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSelected model: {winner}")
    print(f"Saved model: {MODEL_DIR / 'best_model.joblib'}")
    print("\nSUCCESS: Initial model training and validation completed.")
    print("Official test data remains untouched.")


if __name__ == "__main__":
    main()