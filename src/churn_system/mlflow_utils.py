from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

from churn_system.config.config import load_config


def configure_mlflow() -> dict[str, Any]:
    cfg = load_config()
    mcfg = cfg.get("mlflow", {})
    tracking_uri = mcfg.get("tracking_uri", "file:./mlruns")
    experiment_name = mcfg.get("experiment_name", "churn_training")
    registered_model_name = mcfg.get("registered_model_name", "churn_model")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return {
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
    Returns: model URI.
    """
    if tags:
        mlflow.set_tags(tags)

    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )
    return str(model_info.model_uri)


def log_artifact(path: Path) -> None:
    if path.exists():
        mlflow.log_artifact(str(path))

