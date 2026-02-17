"""
Data Drift Detection Module.

Compares training data distribution with
production inference data.
"""

import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("data/training_reference.csv")
PROD_PATH = Path("data/inference_logs/predictions.csv")


NUMERIC_THRESHOLD = 0.2  # mean shift tolerance


def detect_drift():
    """
    Detect simple statistical drift using mean comparison.
    """

    if not TRAIN_PATH.exists() or not PROD_PATH.exists():
        print("Drift check skipped: missing data.")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    prod_df = pd.read_csv(PROD_PATH)

    common_cols = train_df.select_dtypes(include="number").columns

    print("\n=== Drift Report ===")

    for col in common_cols:
        train_mean = train_df[col].mean()
        prod_mean = prod_df[col].mean()

        if train_mean == 0:
            continue

        shift = abs(prod_mean - train_mean) / abs(train_mean)

        if shift > NUMERIC_THRESHOLD:
            print(
                f"⚠ Drift detected in {col} | "
                f"train_mean={train_mean:.2f} "
                f"prod_mean={prod_mean:.2f} "
                f"shift={shift:.2f}"
            )
        else:
            print(f"✓ {col} stable")
