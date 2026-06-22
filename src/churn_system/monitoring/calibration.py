"""
Prediction Calibration and Confidence Monitor.

Industry-level monitoring of model prediction quality:

1. **Calibration Analysis**: Measures how well predicted probabilities
   match actual event rates (using Expected Calibration Error).
2. **Confidence Distribution**: Tracks the distribution of prediction
   confidence to detect model degradation.
3. **Prediction Entropy**: Measures the uncertainty of predictions —
   high entropy indicates the model is uncertain across many requests.
4. **Class Balance Monitoring**: Detects shifts in the predicted class
   distribution compared to training.
5. **Gini Coefficient**: Measures the discriminative power of the model
   from prediction score distributions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger
from churn_system.observability.metrics import (
    CALIBRATION_ERROR,
    PREDICTION_CONFIDENCE_HISTOGRAM,
    PREDICTION_ENTROPY,
)

logger = get_logger(__name__, CONFIG["logging"]["monitoring"])

REPORT_DIR = Path(CONFIG["paths"]["monitoring_dir"])
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_REPORT_FILE = REPORT_DIR / "calibration_report.json"


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute the Expected Calibration Error (ECE).

    ECE is the weighted average of the difference between predicted
    probabilities and actual outcomes within each confidence bin.
    A perfectly calibrated model has ECE = 0.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probabilities for the positive class.
    actuals : np.ndarray
        Binary ground-truth labels (0 or 1).
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    dict
        ECE score and per-bin calibration details.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins_detail = []
    ece = 0.0

    for i in range(n_bins):
        mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
        bin_count = int(mask.sum())

        if bin_count == 0:
            continue

        bin_probs = probabilities[mask]
        bin_actuals = actuals[mask]

        avg_confidence = float(np.mean(bin_probs))
        avg_accuracy = float(np.mean(bin_actuals))
        gap = abs(avg_confidence - avg_accuracy)

        ece += gap * (bin_count / len(probabilities))

        bins_detail.append({
            "bin_range": f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
            "count": bin_count,
            "avg_confidence": round(avg_confidence, 4),
            "avg_accuracy": round(avg_accuracy, 4),
            "calibration_gap": round(gap, 4),
        })

    return {
        "ece": round(float(ece), 6),
        "n_bins": n_bins,
        "bins": bins_detail,
    }


def compute_confidence_distribution(probabilities: np.ndarray) -> dict:
    """
    Analyze the distribution of prediction confidence scores.

    Returns summary statistics and distributional bins for dashboarding.
    """
    confidence = np.maximum(probabilities, 1 - probabilities)

    distribution = {
        "mean_confidence": round(float(np.mean(confidence)), 4),
        "median_confidence": round(float(np.median(confidence)), 4),
        "std_confidence": round(float(np.std(confidence)), 4),
        "p5_confidence": round(float(np.percentile(confidence, 5)), 4),
        "p95_confidence": round(float(np.percentile(confidence, 95)), 4),
        "low_confidence_ratio": round(float((confidence < 0.6).mean()), 4),
        "high_confidence_ratio": round(float((confidence > 0.9).mean()), 4),
    }

    # Record into Prometheus histogram
    for conf in confidence:
        PREDICTION_CONFIDENCE_HISTOGRAM.observe(float(conf))

    return distribution


def compute_prediction_entropy(probabilities: np.ndarray) -> float:
    """
    Compute the average binary entropy of predictions.

    High entropy (close to 1.0) means the model is uncertain —
    many predictions are near 0.5. Low entropy means the model
    is making decisive predictions.

    H(p) = -p*log2(p) - (1-p)*log2(1-p)
    """
    eps = 1e-10
    p = np.clip(probabilities, eps, 1 - eps)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    avg_entropy = float(np.mean(entropy))
    PREDICTION_ENTROPY.set(avg_entropy)
    return round(avg_entropy, 6)


def compute_gini_coefficient(probabilities: np.ndarray) -> float:
    """
    Compute the Gini coefficient of the prediction distribution.

    Gini = 2 * AUC - 1.

    Without ground truth, we approximate Gini from the prediction
    score spread. A Gini close to 0 means no discriminative power.
    """
    sorted_probs = np.sort(probabilities)
    n = len(sorted_probs)
    if n == 0:
        return 0.0

    cumulative = np.cumsum(sorted_probs)
    gini = (2 * np.sum(cumulative) / (n * np.sum(sorted_probs))) - (n + 1) / n
    return round(float(gini), 6)


def compute_class_balance(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    reference_positive_rate: float | None = None,
) -> dict:
    """
    Monitor the predicted class balance.

    Compares the current positive prediction rate to a reference
    baseline. Large deviations suggest concept drift or label shift.
    """
    predicted_positive = float((probabilities >= threshold).mean())
    predicted_negative = 1.0 - predicted_positive

    balance = {
        "predicted_positive_rate": round(predicted_positive, 4),
        "predicted_negative_rate": round(predicted_negative, 4),
        "threshold": threshold,
    }

    if reference_positive_rate is not None:
        shift = abs(predicted_positive - reference_positive_rate)
        balance["reference_positive_rate"] = round(reference_positive_rate, 4)
        balance["class_balance_shift"] = round(shift, 4)
        balance["shift_alert"] = shift > 0.10  # 10% shift threshold

    return balance


def generate_calibration_report(
    probabilities: np.ndarray,
    actuals: np.ndarray | None = None,
    reference_positive_rate: float | None = None,
) -> dict:
    """
    Generate a comprehensive prediction quality report.

    This is the main entry point combining all calibration metrics.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted churn probabilities.
    actuals : np.ndarray, optional
        Ground-truth labels (needed for ECE computation).
    reference_positive_rate : float, optional
        Historical positive class rate for balance monitoring.

    Returns
    -------
    dict
        Full calibration and confidence report.
    """
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(probabilities),
    }

    # Confidence distribution (always available)
    report["confidence_distribution"] = compute_confidence_distribution(probabilities)

    # Prediction entropy
    report["prediction_entropy"] = compute_prediction_entropy(probabilities)

    # Gini coefficient
    report["gini_coefficient"] = compute_gini_coefficient(probabilities)

    # Class balance
    report["class_balance"] = compute_class_balance(
        probabilities,
        threshold=CONFIG["inference"]["threshold"],
        reference_positive_rate=reference_positive_rate,
    )

    # Calibration (only if ground truth is available)
    if actuals is not None:
        calibration = compute_expected_calibration_error(probabilities, actuals)
        report["calibration"] = calibration
        CALIBRATION_ERROR.set(calibration["ece"])
        logger.info("ECE computed | ece=%.6f", calibration["ece"])

    # Persist
    with open(CALIBRATION_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(
        "Calibration report generated | entropy=%.4f | gini=%.4f",
        report["prediction_entropy"],
        report["gini_coefficient"],
    )

    return report
