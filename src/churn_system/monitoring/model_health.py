"""
Model Health Evaluation

Uses drift results to decide whether
model retraining should be triggered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger
from churn_system.monitoring.drift import (
    calculate_psi,
    min_production_samples,
    psi_threshold,
)
from churn_system.observability.metrics import DRIFTING_FEATURES, RETRAINING_RECOMMENDED

logger = get_logger(__name__, CONFIG["logging"]["monitoring"])


def _report_path() -> Path:
    report_dir = Path(CONFIG["paths"]["monitoring_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / "health_report.json"


def _drift_feature_limit() -> int:
    return int(CONFIG.get("monitoring", {}).get("drift_feature_limit", 2))


def _write_report(report: dict[str, Any]) -> dict[str, Any]:
    with open(_report_path(), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    DRIFTING_FEATURES.set(report["drifting_feature_count"])
    RETRAINING_RECOMMENDED.set(1 if report["retraining_recommended"] else 0)

    logger.info("Model health: %s", json.dumps(report))
    return report


def evaluate_model_health() -> dict[str, Any]:
    """
    Evaluate model stability using PSI drift metrics.

    Returns the health report that was written to disk.

    Retraining is only recommended when there is enough production data to make the
    comparison meaningful. PSI against a small sample is enormous regardless of real
    drift — a handful of smoke-test requests would otherwise trigger a full
    retrain-and-promote cycle.
    """
    from churn_system.monitoring.prediction_reader import load_reference_and_production

    frames = load_reference_and_production()
    if frames is None:
        logger.warning(
            "Health evaluation skipped: reference or production data unavailable."
        )
        return _write_report(
            {
                "status": "insufficient_data",
                "reason": "missing reference or production data",
                "drifting_feature_count": 0,
                "drifting_features": [],
                "retraining_recommended": False,
            }
        )

    train_df, prod_df = frames

    threshold = psi_threshold()
    minimum = min_production_samples()

    numeric_cols = train_df.select_dtypes(include=np.number).columns

    drifting_features: list[dict[str, Any]] = []
    evaluated: list[str] = []
    skipped: list[str] = []

    for col in numeric_cols:
        if col not in prod_df.columns:
            continue

        train_series = train_df[col].dropna()
        prod_series = prod_df[col].dropna()

        if len(prod_series) < minimum or train_series.empty:
            skipped.append(str(col))
            continue

        psi = calculate_psi(train_series, prod_series)
        evaluated.append(str(col))

        if psi > threshold:
            drifting_features.append({"feature": str(col), "psi": round(float(psi), 4)})

    if not evaluated:
        return _write_report(
            {
                "status": "insufficient_data",
                "reason": (
                    f"no feature had at least {minimum} production samples "
                    f"(skipped: {', '.join(skipped) or 'none'})"
                ),
                "drifting_feature_count": 0,
                "drifting_features": [],
                "evaluated_features": [],
                "skipped_features": skipped,
                "retraining_recommended": False,
            }
        )

    retrain_required = len(drifting_features) >= _drift_feature_limit()

    return _write_report(
        {
            "status": "evaluated",
            "drifting_feature_count": len(drifting_features),
            "drifting_features": drifting_features,
            "evaluated_features": evaluated,
            "skipped_features": skipped,
            "psi_threshold": threshold,
            "drift_feature_limit": _drift_feature_limit(),
            "retraining_recommended": retrain_required,
        }
    )


if __name__ == "__main__":
    evaluate_model_health()
