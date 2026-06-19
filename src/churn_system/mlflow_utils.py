"""MLflow integration helpers with retry logic for transient network failures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

from churn_system.config.config import load_config
from churn_system.utils.retry import retry_with_backoff


def configure_mlflow() -> dict[str, Any]:
    """Configure MLflow tracking URI and experiment. Returns config dict."""
    cfg = load_config()
    mcfg = cfg.get("mlflow", {})
    enabled = os.environ.get("CHURN_MLFLOW_ENABLED", "1").lower() not in {"0", "false", "no"}
    tracking_uri = os.environ.get("CHURN_MLFLOW_TRACKING_URI") or mcfg.get(
        "tracking_uri", "file:./mlruns"
    )
    experiment_name = mcfg.get("experiment_name", "churn_training")
    registered_model_name = mcfg.get("registered_model_name", "churn_model")

    if not enabled:
        return {
            "enabled": False,
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "registered_model_name": registered_model_name,
        }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return {
        "enabled": True,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
    }


def log_sklearn_model(
    *,
    pipeline,
    registered_model_name: str,
    artifact_path: str = "model",
    tags: dict[str, str] | None = None,
) -> str:
    """
    Log and (optionally) register model with MLflow Model Registry.

    Retries on transient network/DB errors.

    Returns: model URI.
    """
    if tags:
        mlflow.set_tags(tags)

    def _log():
        return mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )

    model_info = retry_with_backoff(
        _log,
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(ConnectionError, OSError, Exception),
        operation_name="mlflow.log_model",
    )
    return str(model_info.model_uri)


def log_artifact(path: Path) -> None:
    """Log a file artifact to MLflow with retry."""
    if not path.exists():
        return

    def _log():
        mlflow.log_artifact(str(path))

    retry_with_backoff(
        _log,
        max_retries=2,
        base_delay=0.5,
        retryable_exceptions=(ConnectionError, OSError, Exception),
        operation_name="mlflow.log_artifact",
    )
