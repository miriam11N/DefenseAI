from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "FD001"
SPLIT_DIR = DATA_DIR / "splits"
FIGURE_DIR = ROOT / "reports" / "figures" / "FD001"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42

CANDIDATE_FEATURES = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def save_figure(filename):
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[CHART] {filename}")


def main():
    for directory in [SPLIT_DIR, FIGURE_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    source = DATA_DIR / "train.csv"

    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Run prepare_fd001.py first."
        )

    data = pd.read_csv(source)

    required = ["engine_id", "cycle", "rul"] + CANDIDATE_FEATURES
    missing = sorted(set(required) - set(data.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data[required].isna().any().any():
        raise ValueError("Missing values detected.")

    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values detected.")

    if data.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle records detected.")

    if data["engine_id"].nunique() != 100:
        raise ValueError("Expected 100 engines in the source training data.")

    data = data.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_indices, validation_indices = next(
        splitter.split(data, groups=data["engine_id"])
    )

    train = data.iloc[train_indices].copy().reset_index(drop=True)
    validation = (
        data.iloc[validation_indices].copy().reset_index(drop=True)
    )

    train_ids = sorted(train["engine_id"].unique().tolist())
    validation_ids = sorted(validation["engine_id"].unique().tolist())

    overlap = sorted(set(train_ids) & set(validation_ids))

    if overlap:
        raise ValueError(f"Engine overlap detected: {overlap}")

    if len(train_ids) != 80 or len(validation_ids) != 20:
        raise ValueError("Expected an 80-engine / 20-engine split.")

    if len(train) + len(validation) != len(data):
        raise ValueError("The split did not preserve every source row.")

    # Feature selection is based only on the training partition.
    constant_features = [
        column
        for column in CANDIDATE_FEATURES
        if train[column].nunique() <= 1
    ]

    feature_columns = [
        column
        for column in CANDIDATE_FEATURES
        if column not in constant_features
    ]

    # Keep identifiers and labels in the CSVs for auditing.
    # Future models must use only feature_columns as model inputs.
    train.to_csv(SPLIT_DIR / "train_split.csv", index=False)
    validation.to_csv(
        SPLIT_DIR / "validation_split.csv", index=False
    )

    config = {
        "dataset": "FD001",
        "random_state": RANDOM_STATE,
        "split_method": "GroupShuffleSplit by engine_id",
        "train_engine_ids": train_ids,
        "validation_engine_ids": validation_ids,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "feature_columns": feature_columns,
        "dropped_constant_features": constant_features,
        "target_column": "rul",
        "target_units": "operating cycles",
        "target_capping": None,
        "excluded_model_inputs": ["engine_id", "cycle", "rul"],
        "official_test_used": False,
    }

    (SPLIT_DIR / "split_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    train.describe().to_csv(
        REPORT_DIR / "fd001_training_statistics.csv"
    )

    # All exploratory plots below use training-partition data only.
    lifetimes = train.groupby("engine_id")["cycle"].max()

    plt.figure(figsize=(9, 5))
    plt.hist(
        lifetimes,
        bins=15,
        color="steelblue",
        edgecolor="white",
    )
    plt.title("FD001: Training-Partition Engine Lifetimes")
    plt.xlabel("Operating cycles until failure")
    plt.ylabel("Number of engines")
    save_figure("01_engine_lifetimes.png")

    plt.figure(figsize=(9, 5))
    plt.hist(
        train["rul"],
        bins=30,
        color="darkorange",
        edgecolor="white",
    )
    plt.title("FD001: Training-Partition RUL Distribution")
    plt.xlabel("Remaining useful life (cycles)")
    plt.ylabel("Number of sensor records")
    save_figure("02_rul_distribution.png")

    selected_engines = train_ids[:5]

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 10), sharex=True
    )

    for axis, sensor in zip(
        axes, ["sensor_2", "sensor_7", "sensor_11"]
    ):
        for engine_id in selected_engines:
            engine = train.loc[train["engine_id"] == engine_id]
            axis.plot(
                engine["cycle"],
                engine[sensor],
                label=f"Engine {engine_id}",
                linewidth=1,
                alpha=0.8,
            )

        axis.set_ylabel(sensor)
        axis.grid(alpha=0.25)

    axes[0].legend(ncol=5, fontsize=8)
    axes[0].set_title(
        "FD001: Example Training-Engine Sensor Trajectories"
    )
    axes[-1].set_xlabel("Operating cycle")
    save_figure("03_sensor_trends.png")

    correlations = (
        train[feature_columns]
        .corrwith(train["rul"])
        .dropna()
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    colors = [
        "steelblue" if value < 0 else "darkorange"
        for value in correlations
    ]
    plt.barh(
        correlations.index,
        correlations.values,
        color=colors,
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlim(-1, 1)
    plt.xlabel("Pearson correlation with RUL")
    plt.title(
        "FD001: Training-Partition Feature Correlations"
    )
    save_figure("04_feature_rul_correlations.png")

    print("\n" + "=" * 65)
    print("SPLIT SUMMARY")
    print("=" * 65)
    print(f"Training engines:        {len(train_ids)}")
    print(f"Validation engines:      {len(validation_ids)}")
    print(f"Training rows:           {len(train):,}")
    print(f"Validation rows:         {len(validation):,}")
    print(f"Overlapping engines:     {len(overlap)}")
    print(f"Selected model features: {len(feature_columns)}")
    print(f"Constant features:       {constant_features}")

    print("\nSelected features:")
    print(feature_columns)

    print("\nValidation engine IDs:")
    print(validation_ids)

    print(f"\nSplit files: {SPLIT_DIR}")
    print(f"Charts:      {FIGURE_DIR}")
    print("\nSUCCESS: Analysis and engine-level split completed.")
    print("Official test data was not loaded or modified.")


if __name__ == "__main__":
    main()