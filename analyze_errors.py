from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "FD001"
MODEL_DIR = ROOT / "artifacts" / "FD001"


def summarize(frame):
    errors = frame["error_cycles"].to_numpy()

    return {
        "records": len(frame),
        "mae_cycles": float(np.mean(np.abs(errors))),
        "rmse_cycles": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error_cycles": float(np.mean(errors)),
        "overestimate_percent": float(np.mean(errors > 0) * 100),
    }


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()
    print(f"[CHART] {filename}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    summary = json.loads(
        (MODEL_DIR / "training_summary.json").read_text(
            encoding="utf-8"
        )
    )

    winner = summary["selected_model"]
    prediction_column = f"{winner}_prediction"

    comparison = pd.read_csv(
        REPORT_DIR / "fd001_model_comparison.csv"
    )

    predictions = pd.read_csv(
        REPORT_DIR / "fd001_validation_predictions.csv"
    )

    required = ["engine_id", "cycle", "rul", prediction_column]

    if not set(required).issubset(predictions.columns):
        raise ValueError("Required prediction columns are missing.")

    if not np.isfinite(
        predictions[required].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid numeric values in predictions.")

    if predictions.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle records.")

    predictions = predictions.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    # Positive error means the model predicts too much remaining life.
    predictions["error_cycles"] = (
        predictions[prediction_column] - predictions["rul"]
    )
    predictions["absolute_error_cycles"] = (
        predictions["error_cycles"].abs()
    )

    overall = summarize(predictions)

    engine_rows = []

    for engine_id, group in predictions.groupby("engine_id"):
        engine_rows.append({
            "engine_id": int(engine_id),
            **summarize(group),
        })

    engine_metrics = (
        pd.DataFrame(engine_rows)
        .sort_values("mae_cycles", ascending=False)
        .reset_index(drop=True)
    )

    # These are descriptive analysis bands, not maintenance thresholds.
    bands = [
        ("0-30", predictions["rul"].between(0, 30)),
        ("31-60", predictions["rul"].between(31, 60)),
        ("61-120", predictions["rul"].between(61, 120)),
        ("121+", predictions["rul"] > 120),
    ]

    band_rows = []

    for label, mask in bands:
        group = predictions.loc[mask]
        if not group.empty:
            band_rows.append({
                "actual_rul_band": label,
                **summarize(group),
            })

    band_metrics = pd.DataFrame(band_rows)

    # Every validation engine contributes equally to this mean of MAEs.
    macro_engine_mae = float(engine_metrics["mae_cycles"].mean())

    predictions.to_csv(
        REPORT_DIR / "fd001_validation_error_details.csv",
        index=False,
    )
    engine_metrics.to_csv(
        REPORT_DIR / "fd001_errors_by_engine.csv",
        index=False,
    )
    band_metrics.to_csv(
        REPORT_DIR / "fd001_errors_by_rul_band.csv",
        index=False,
    )

    error_summary = {
        "model": winner,
        "population": "All cycles from held-out validation engines",
        "error_definition": "predicted RUL minus actual RUL",
        "overall": overall,
        "equal_engine_weighted_mae_cycles": macro_engine_mae,
        "official_test_used": False,
        "notes": [
            "Positive error means remaining life is overestimated.",
            "RUL bands use ground truth for retrospective analysis only.",
            "RUL bands are not operational maintenance thresholds.",
            "Validation contains complete run-to-failure trajectories.",
            "No models were retrained.",
        ],
    }

    (REPORT_DIR / "fd001_error_summary.json").write_text(
        json.dumps(error_summary, indent=2),
        encoding="utf-8",
    )

    plt.figure(figsize=(9, 5))
    plt.hist(
        predictions["error_cycles"],
        bins=50,
        color="steelblue",
        edgecolor="white",
    )
    plt.axvline(
        0, color="black", linestyle="--", label="Zero error"
    )
    plt.xlabel("Predicted RUL - actual RUL (cycles)")
    plt.ylabel("Number of records")
    plt.title(f"{winner}: Validation Error Distribution")
    plt.legend()
    save_chart("08_error_distribution.png")

    plt.figure(figsize=(9, 5))
    plt.bar(
        band_metrics["actual_rul_band"],
        band_metrics["mae_cycles"],
        color="darkorange",
    )
    plt.xlabel("Actual remaining useful life (cycles)")
    plt.ylabel("MAE (cycles)")
    plt.title(f"{winner}: Error by Actual Remaining-Life Range")
    save_chart("09_error_by_rul_band.png")

    plt.figure(figsize=(11, 5))
    plt.bar(
        engine_metrics["engine_id"].astype(str),
        engine_metrics["mae_cycles"],
        color="steelblue",
    )
    plt.xlabel("Validation engine ID")
    plt.ylabel("MAE (cycles)")
    plt.title(f"{winner}: Per-Engine Error, Worst to Best")
    save_chart("10_error_by_engine.png")

    # Show the three highest-MAE engines and the lowest-MAE engine.
    selected_engines = (
        engine_metrics.head(3)["engine_id"].tolist()
        + engine_metrics.tail(1)["engine_id"].tolist()
    )
    selected_engines = list(dict.fromkeys(selected_engines))

    fig, axes = plt.subplots(
        len(selected_engines),
        1,
        figsize=(11, 3 * len(selected_engines)),
        squeeze=False,
    )

    for axis, engine_id in zip(axes.ravel(), selected_engines):
        group = predictions.loc[
            predictions["engine_id"] == engine_id
        ]

        axis.plot(
            group["cycle"],
            group["rul"],
            label="Actual RUL",
            linewidth=2,
        )
        axis.plot(
            group["cycle"],
            group[prediction_column],
            label="Predicted RUL",
            alpha=0.85,
        )

        axis.set_title(f"Validation Engine {engine_id}")
        axis.set_xlabel("Operating cycle")
        axis.set_ylabel("RUL (cycles)")
        axis.grid(alpha=0.25)
        axis.legend()

    save_chart("11_validation_engine_trajectories.png")

    print("\n" + "=" * 70)
    print("SAVED MODEL COMPARISON")
    print("=" * 70)
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print(f"\nSelected model: {winner}")

    print("\nOVERALL VALIDATION ERRORS")
    for key, value in overall.items():
        print(f"{key}: {value:.2f}")

    print(f"Equal-engine-weighted MAE: {macro_engine_mae:.2f}")

    print("\nERROR BY ACTUAL RUL BAND")
    print(
        band_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\nFIVE HIGHEST-MAE ENGINES")
    print(
        engine_metrics.head(5).to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\nSUCCESS: Validation error analysis completed.")
    print("No retraining. Official test data remains untouched.")


if __name__ == "__main__":
    main()