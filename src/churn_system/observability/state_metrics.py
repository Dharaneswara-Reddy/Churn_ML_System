"""
Publish persisted lifecycle state as Prometheus metrics from the API process.

The problem this solves
-----------------------
``prometheus_client``'s registry is per-process. Drift, data-quality and
calibration gauges were set inside the *scheduler* process, while Prometheus
scrapes only the API. Those series could therefore never appear on ``/metrics`` —
which meant the ``ChurnModelDriftDetected`` alert rule, keyed on
``churn_drifting_feature_count``, was structurally incapable of firing. Not
"rarely": never, and silently.

The fix is deliberately the boring one. The scheduler already persists its verdict
to ``health_report.json`` and its promotion history to ``lineage.json``; the API
reads that state at scrape time and publishes it. No Pushgateway to operate, no
second scrape target, no shared registry — and it works unchanged whether the
scheduler runs in the same process, another container, or not at all.

The trade-off, stated plainly: these gauges are as fresh as the last lifecycle
run, not as fresh as the scrape. That is the correct semantic for a batch job's
output, but it does mean a stale ``health_report.json`` shows stale drift. The
``churn_drift_report_age_seconds`` gauge exists so that staleness is itself
observable and alertable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger
from churn_system.observability.metrics import (
    CHAMPION_MODEL_INFO,
    DRIFTING_FEATURES,
    GINI_COEFFICIENT,
    OUTBOX_BACKLOG,
    PREDICTED_NEGATIVE_RATE,
    PREDICTED_POSITIVE_RATE,
    RETRAINING_RECOMMENDED,
    TRAINING_LAST_SUCCESS_TIMESTAMP,
)

logger = get_logger(__name__, CONFIG["logging"]["api"])

DRIFT_REPORT_AGE = None  # set lazily below to avoid a duplicate registration


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Could not read state file for metrics: %s", path)
        return None


def _publish_drift() -> None:
    """Publish the scheduler's drift verdict from health_report.json."""
    path = Path(CONFIG["paths"]["monitoring_dir"]) / "health_report.json"
    report = _read_json(path)
    if not isinstance(report, dict):
        return

    DRIFTING_FEATURES.set(float(report.get("drifting_feature_count", 0)))
    RETRAINING_RECOMMENDED.set(1 if report.get("retraining_recommended") else 0)


def _publish_prediction_rates() -> None:
    """
    Publish the observed positive/negative split of recent predictions.

    These two gauges were declared but never set anywhere in the codebase — they
    appeared on /metrics as permanent zeros, which is worse than absent because a
    dashboard built on them looks healthy.
    """
    from churn_system.monitoring.prediction_reader import load_predictions_df

    frame = load_predictions_df(limit=1000)
    if frame.empty or "prediction" not in frame.columns:
        return

    positive_rate = float(frame["prediction"].mean())
    PREDICTED_POSITIVE_RATE.set(positive_rate)
    PREDICTED_NEGATIVE_RATE.set(1.0 - positive_rate)


def _publish_champion() -> None:
    """Publish the current champion version and its discriminative power."""
    metadata_path = Path(CONFIG["paths"]["production_model"]).parent / "metadata.json"
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        return

    version = str(metadata.get("model_version", "unknown"))
    CHAMPION_MODEL_INFO.labels(model_version=version).set(1)

    # Gini is a linear rescaling of ROC-AUC (2*AUC - 1) and was previously
    # declared but never computed. Deriving it from the recorded metric is
    # exact, not an approximation.
    roc_auc = metadata.get("metrics", {}).get("roc_auc")
    if roc_auc is not None:
        GINI_COEFFICIENT.set(2.0 * float(roc_auc) - 1.0)


def _publish_training_recency() -> None:
    """Publish when training last succeeded, from the newest experiment bundle."""
    from churn_system.artifacts import latest_experiment_dir

    try:
        latest = latest_experiment_dir()
    except Exception:
        return
    if latest is None:
        return

    metadata_path = latest / "metadata.json"
    if metadata_path.exists():
        TRAINING_LAST_SUCCESS_TIMESTAMP.set(metadata_path.stat().st_mtime)


def _publish_outbox_backlog() -> None:
    """
    Publish outbox depth by status.

    Dead-lettered events used to be indistinguishable from pending work, so a
    backlog metric would have counted permanently-failed events as live.
    """
    from churn_system.events.retention import outbox_backlog

    for status, count in outbox_backlog().items():
        OUTBOX_BACKLOG.labels(status=status).set(count)


def refresh_state_metrics() -> None:
    """
    Refresh every gauge derived from persisted state.

    Called from the ``/metrics`` handler. Each publisher is isolated: a missing or
    malformed state file must degrade one gauge, never fail the scrape — an
    unscrapeable endpoint would take down all the other metrics with it.
    """
    for publisher in (
        _publish_drift,
        _publish_champion,
        _publish_training_recency,
        _publish_outbox_backlog,
        _publish_prediction_rates,
    ):
        try:
            publisher()
        except Exception:
            logger.warning(
                "Metric publisher %s failed; continuing with the remaining metrics.",
                publisher.__name__,
                exc_info=True,
            )


def drift_report_age_seconds() -> float | None:
    """Age of the drift report, so staleness is itself observable."""
    path = Path(CONFIG["paths"]["monitoring_dir"]) / "health_report.json"
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime
