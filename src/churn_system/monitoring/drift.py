"""
Data Drift Detection Module

Compares training data distribution with production
inference data using Population Stability Index (PSI).

PSI measures how much a feature's distribution has
shifted between training and production data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


TRAIN_PATH = Path("data/training_reference.csv")
PROD_PATH = Path("data/inference_logs/predictions.csv")


PSI_THRESHOLD = 0.2



def calculate_psi(expected: pd.Series,
                  actual: pd.Series,
                  bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).

    Parameters
    ----------
    expected : pd.Series
        Training distribution (reference).
    actual : pd.Series
        Production distribution.
    bins : int
        Number of histogram bins.

    Returns
    -------
    float
        PSI score.
    """

    expected = expected.values
    actual = actual.values

    expected_counts, bin_edges = np.histogram(expected, bins=bins)

    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    psi_values = []

    for e, a in zip(expected_percents, actual_percents):
        # smoothing to avoid log(0)
        e = max(e, 1e-6)
        a = max(a, 1e-6)

        psi_values.append((a - e) * np.log(a / e))

    return float(np.sum(psi_values))



def detect_drift() -> None:
    """
    Compare training and production datasets and
    report feature-level drift.
    """

    if not TRAIN_PATH.exists() or not PROD_PATH.exists():
        print(" Missing training or production data.")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    prod_df = pd.read_csv(PROD_PATH)

    numeric_cols = train_df.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        print(" No numeric columns found for drift detection.")
        return

    print("\n----------- PSI Drift Report -----------")

    for col in numeric_cols:

        if col not in prod_df.columns:
            continue

        train_series = train_df[col].dropna()
        prod_series = prod_df[col].dropna()

        # Skip if insufficient production data
        if len(prod_series) < 20:
            print(f"{col:<22} |  insufficient production samples")
            continue

        psi = calculate_psi(train_series, prod_series)

        status = " DRIFT" if psi > PSI_THRESHOLD else " STABLE"

        print(f"{col:<22} | {status} | PSI={psi:.4f}")

    print("-----------------------------------------------\n")


# CLI entry
if __name__ == "__main__":
    detect_drift()
