"""
Data Drift Detection Module

Compares training data distribution with production
inference data using Population Stability Index (PSI).

PSI measures how much a feature's distribution has
shifted between training and production data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["monitoring"])


def psi_threshold() -> float:
    return float(CONFIG.get("monitoring", {}).get("psi_threshold", 0.2))


def psi_bins() -> int:
    return int(CONFIG.get("monitoring", {}).get("psi_bins", 10))


def min_production_samples() -> int:
    return int(CONFIG.get("monitoring", {}).get("min_production_samples", 20))


# Kept for backwards compatibility with callers that imported the constant.
PSI_THRESHOLD = psi_threshold()


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int | None = None,
) -> float:
    """
    Compute Population Stability Index (PSI).

    Parameters
    ----------
    expected : pd.Series
        Training distribution (reference).
    actual : pd.Series
        Production distribution.
    bins : int, optional
        Number of interior histogram bins. Defaults to the configured value.

    Returns
    -------
    float
        PSI score. Always finite and non-negative.

    Raises
    ------
    ValueError
        If either series is empty after dropping NaNs. An empty production feed is
        a monitoring failure, not a stable distribution — returning NaN here would
        make every downstream ``psi > threshold`` comparison silently False.

    Notes
    -----
    Two corrections relative to a naive implementation:

    * **Out-of-range mass is counted, not dropped.** ``np.histogram`` with fixed
      edges discards values outside the reference range; dividing by the full
      sample length then spreads that loss across every bin and *understates*
      drift. Explicit underflow/overflow bins keep both distributions summing to
      one, so production data that has moved off the reference range registers as
      the severe drift it is.
    * **Counts get additive (Laplace) smoothing** instead of flooring already
      normalised proportions, which otherwise turns a single sparse reference bin
      into a large spurious contribution.
    """
    bin_count = psi_bins() if bins is None else bins

    expected_values = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual_values = pd.Series(actual).dropna().to_numpy(dtype=float)

    if expected_values.size == 0 or actual_values.size == 0:
        raise ValueError(
            "PSI requires non-empty reference and production samples "
            f"(reference={expected_values.size}, production={actual_values.size})"
        )

    expected_counts, bin_edges = np.histogram(expected_values, bins=bin_count)
    actual_counts, _ = np.histogram(actual_values, bins=bin_edges)

    # Underflow / overflow buckets for values outside the reference range.
    below = int((actual_values < bin_edges[0]).sum())
    above = int((actual_values > bin_edges[-1]).sum())

    expected_counts = np.concatenate(([0], expected_counts, [0])).astype(float)
    actual_counts = np.concatenate(([below], actual_counts, [above])).astype(float)

    # Additive smoothing so an empty bin on either side stays finite.
    alpha = 0.5
    n_bins = expected_counts.size
    expected_percents = (expected_counts + alpha) / (expected_counts.sum() + alpha * n_bins)
    actual_percents = (actual_counts + alpha) / (actual_counts.sum() + alpha * n_bins)

    psi_values = (actual_percents - expected_percents) * np.log(
        actual_percents / expected_percents
    )

    return float(np.sum(psi_values))


def detect_drift() -> None:
    """
    Compare training and production datasets and report feature-level drift.
    """
    from churn_system.monitoring.prediction_reader import load_reference_and_production

    frames = load_reference_and_production()
    if frames is None:
        logger.warning("Drift report skipped: reference or production data unavailable.")
        return

    train_df, prod_df = frames

    numeric_cols = train_df.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found for drift detection.")
        return

    threshold = psi_threshold()
    minimum = min_production_samples()

    logger.info("----------- PSI Drift Report -----------")

    for col in numeric_cols:
        if col not in prod_df.columns:
            continue

        train_series = train_df[col].dropna()
        prod_series = prod_df[col].dropna()

        if len(prod_series) < minimum:
            logger.info(
                "%-22s | insufficient production samples (%d < %d)",
                col,
                len(prod_series),
                minimum,
            )
            continue

        psi = calculate_psi(train_series, prod_series)
        status = "DRIFT" if psi > threshold else "STABLE"

        logger.info("%-22s | %-6s | PSI=%.4f", col, status, psi)

    logger.info("-----------------------------------------")


# CLI entry
if __name__ == "__main__":
    detect_drift()
