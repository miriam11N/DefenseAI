from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "FD001"
PROCESSED_DIR = ROOT / "data" / "processed" / "FD001"
REPORT_DIR = ROOT / "reports"

BASE_URL = (
    "https://huggingface.co/datasets/"
    "SoyVitou/NASA-C-MAPSS-Turbofan-Engine/resolve/main/data"
)

FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
]

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

EXPECTED_ROWS = {
    "train": 20631,
    "test": 13096,
}


def download_file(filename):
    destination = RAW_DIR / filename
    url = f"{BASE_URL}/{filename}"

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[EXISTS] {filename}")
        return destination

    print(f"[DOWNLOADING] {filename}")

    temporary = destination.with_suffix(".download")

    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()

            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        output.write(chunk)

        if temporary.stat().st_size == 0:
            raise ValueError(f"Downloaded file is empty: {filename}")

        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"[SAVED] {destination.name}")
    return destination


def load_sensor_file(path, split):
    frame = pd.read_csv(path, sep=r"\s+", header=None)

    if frame.shape[1] != 26:
        raise ValueError(
            f"{path.name}: expected 26 columns, found {frame.shape[1]}."
        )

    frame.columns = COLUMNS
    frame = frame.apply(pd.to_numeric, errors="raise")

    if frame.isna().any().any():
        raise ValueError(f"{path.name}: missing values detected.")

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: non-finite values detected.")

    for column in ["engine_id", "cycle"]:
        if not (frame[column] == np.floor(frame[column])).all():
            raise ValueError(f"{path.name}: {column} must contain integers.")
        frame[column] = frame[column].astype(int)

    if len(frame) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_ROWS[split]} rows, "
            f"found {len(frame)}."
        )

    if sorted(frame["engine_id"].unique().tolist()) != list(range(1, 101)):
        raise ValueError(f"{path.name}: expected engine IDs 1 through 100.")

    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError(f"{path.name}: duplicate engine-cycle records.")

    frame = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    for engine_id, group in frame.groupby("engine_id"):
        expected_cycles = np.arange(1, len(group) + 1)
        if not np.array_equal(group["cycle"].to_numpy(), expected_cycles):
            raise ValueError(
                f"{path.name}: invalid cycle sequence for engine {engine_id}."
            )

    return frame


def load_test_targets(path):
    targets = pd.read_csv(path, sep=r"\s+", header=None)

    if targets.shape != (100, 1):
        raise ValueError(
            f"{path.name}: expected 100 rows and 1 column, "
            f"found {targets.shape}."
        )

    values = pd.to_numeric(targets.iloc[:, 0], errors="raise").to_numpy()

    if not np.isfinite(values).all():
        raise ValueError("Test RUL targets contain missing or infinite values.")

    if (values < 0).any() or not (values == np.floor(values)).all():
        raise ValueError("Test RUL targets must be non-negative integers.")

    return pd.DataFrame({
        "engine_id": np.arange(1, 101),
        "rul": values.astype(int),
    })


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for directory in [RAW_DIR, PROCESSED_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("DefenseAI: FD001 download and validation")
    print("=" * 65)

    for filename in FILES:
        download_file(filename)

    train = load_sensor_file(RAW_DIR / "train_FD001.txt", "train")
    test = load_sensor_file(RAW_DIR / "test_FD001.txt", "test")
    test_targets = load_test_targets(RAW_DIR / "RUL_FD001.txt")

    # Training trajectories run until failure.
    final_train_cycle = train.groupby("engine_id")["cycle"].transform("max")
    train["rul"] = final_train_cycle - train["cycle"]

    if train["rul"].lt(0).any():
        raise ValueError("Training RUL cannot be negative.")

    if not train.groupby("engine_id")["rul"].last().eq(0).all():
        raise ValueError("Every training engine must end with RUL = 0.")

    # Official test evaluation uses each engine's last observed cycle.
    # Test labels remain separate from the test sensor features.
    test_last = (
        test.groupby("engine_id", sort=True)
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    engine_summary = (
        train.groupby("engine_id", as_index=False)
        .agg(
            observed_cycles=("cycle", "max"),
            initial_rul=("rul", "max"),
        )
    )

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    test_last.to_csv(PROCESSED_DIR / "test_last_cycle.csv", index=False)
    test_targets.to_csv(PROCESSED_DIR / "test_targets.csv", index=False)
    engine_summary.to_csv(
        PROCESSED_DIR / "training_engine_summary.csv", index=False
    )

    constant_features = [
        column
        for column in COLUMNS[2:]
        if train[column].nunique() == 1
    ]

    report = {
        "dataset": "NASA C-MAPSS FD001",
        "data_type": "Simulated turbofan engine degradation",
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_source_type": "Third-party raw-file mirror",
        "files": [
            {
                "filename": filename,
                "source_url": f"{BASE_URL}/{filename}",
                "sha256": sha256(RAW_DIR / filename),
                "bytes": (RAW_DIR / filename).stat().st_size,
            }
            for filename in FILES
        ],
        "train_rows": len(train),
        "test_rows": len(test),
        "train_engines": int(train["engine_id"].nunique()),
        "test_engines": int(test["engine_id"].nunique()),
        "raw_sensor_file_columns": len(COLUMNS),
        "training_rul_min": int(train["rul"].min()),
        "training_rul_max": int(train["rul"].max()),
        "constant_training_features": constant_features,
        "checks_passed": [
            "Expected row and column counts",
            "Numeric and finite measurements",
            "Engine IDs 1 through 100",
            "Unique engine-cycle pairs",
            "Consecutive cycles starting at 1",
            "100 non-negative integer test RUL labels",
            "Training trajectories end at RUL zero",
        ],
        "notes": [
            "RUL is measured in operating cycles, not hours.",
            "Training RUL is uncapped.",
            "Test targets apply only to each engine's last observed cycle.",
            "Training and test engine IDs are split-local identifiers.",
            "No scaling, feature selection, or model training performed.",
            "File hashes record local content; they do not certify NASA authenticity.",
        ],
    }

    report_path = REPORT_DIR / "fd001_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nValidation summary")
    print("-" * 65)
    print(f"Training rows:    {len(train):,}")
    print(f"Test rows:        {len(test):,}")
    print(f"Training engines: {train['engine_id'].nunique()}")
    print(f"Test engines:     {test['engine_id'].nunique()}")
    print(f"Test targets:     {len(test_targets)}")
    print(f"Training RUL:     {train['rul'].min()} to {train['rul'].max()} cycles")
    print(f"Constant features: {constant_features}")

    print("\nTraining preview")
    print(
        train[
            ["engine_id", "cycle", "sensor_2", "sensor_3", "rul"]
        ].head().to_string(index=False)
    )

    print(f"\nProcessed files: {PROCESSED_DIR}")
    print(f"Validation report: {report_path}")
    print("\nSUCCESS: FD001 downloaded, validated, and prepared.")


if __name__ == "__main__":
    main()