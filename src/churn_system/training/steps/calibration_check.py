"""
Calibration measurement.

A model can rank well and still be badly calibrated. This one was: measured on
the old tenure split it predicted a mean probability of 0.1035 against an actual
positive rate of 0.0660 — over-predicting by 1.57x — because it was calibrated to
the *training* fold's 31.5% prevalence while being scored on a 6.6% fold.

``/predict`` returns a probability that downstream consumers will read as a
probability, so this is a correctness property, not a reporting nicety.

Measure first, correct only if the measurement warrants it. ``needs_calibration``
applies an explicit, configurable tolerance rather than reflexively wrapping every
model in ``CalibratedClassifierCV`` — an unnecessary calibration layer costs an
extra fit, obscures the base estimator, and can degrade a model that was already
well calibrated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["training"])

# Relative tolerance on mean-predicted vs actual rate before calibration is
# considered necessary. 0.25 means "within 25% of the true positive rate".
DEFAULT_TOLERANCE = 0.25


def _tolerance() -> float:
    return float(
        CONFIG.get("calibration", {}).get("relative_tolerance", DEFAULT_TOLERANCE)
    )


def reliability_curve(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 5
) -> list[dict[str, Any]]:
    """
    Bin predictions by predicted probability and report the actual rate in each.

    Uses quantile bins so every bin holds a comparable number of rows; equal-width
    bins on a skewed probability distribution leave most bins nearly empty.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) == 0:
        return []

    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return [
            {
                "bin": 0,
                "n": len(y_true),
                "mean_predicted": float(probabilities.mean()),
                "actual_rate": float(y_true.mean()),
            }
        ]

    indices = np.clip(np.digitize(probabilities, edges[1:-1]), 0, len(edges) - 2)
    curve = []
    for b in range(len(edges) - 1):
        mask = indices == b
        if not mask.any():
            continue
        curve.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_predicted": float(probabilities[mask].mean()),
                "actual_rate": float(y_true[mask].mean()),
            }
        )
    return curve


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 5
) -> float:
    """Weighted mean absolute gap between predicted and actual rate per bin."""
    curve = reliability_curve(y_true, probabilities, bins=bins)
    if not curve:
        return float("nan")
    total = sum(row["n"] for row in curve)
    return float(
        sum(
            row["n"] * abs(row["mean_predicted"] - row["actual_rate"])
            for row in curve
        )
        / total
    )


def measure_calibration(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 5
) -> dict[str, Any]:
    """
    Describe how well predicted probabilities match observed frequencies.

    Must be given validation data — measuring calibration on the data the model
    was fitted on reports the fit, not the generalisation.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    actual_rate = float(y_true.mean())
    mean_predicted = float(probabilities.mean())
    ratio = float(mean_predicted / actual_rate) if actual_rate > 0 else float("nan")

    return {
        "mean_predicted_probability": mean_predicted,
        "actual_positive_rate": actual_rate,
        "calibration_ratio": ratio,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities, bins=bins
        ),
        "reliability_curve": reliability_curve(y_true, probabilities, bins=bins),
        "n": len(y_true),
    }


def needs_calibration(report: dict[str, Any], tolerance: float | None = None) -> bool:
    """
    Decide whether a calibration correction is warranted.

    True when the mean predicted probability deviates from the observed positive
    rate by more than ``tolerance`` in relative terms.
    """
    limit = _tolerance() if tolerance is None else tolerance
    ratio = report.get("calibration_ratio")
    if ratio is None or not np.isfinite(ratio):
        return False
    return abs(ratio - 1.0) > limit
