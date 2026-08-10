"""Promotion of a trained experiment bundle into the production serving slot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from churn_system.artifacts import load_metadata, swap_model_bundle
from churn_system.config.config import CONFIG
from churn_system.lifecycle.lineage import record_lineage
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])


def schemas_match(prod_meta: dict[str, Any], new_meta: dict[str, Any]) -> bool:
    prod_schema = prod_meta.get("feature_schema", [])
    new_schema = new_meta.get("feature_schema", [])

    return prod_schema == new_schema


def promote_model(version: str) -> bool:
    """
    Promote a trained model version to production.

    Ensures schema compatibility before promotion, and swaps the bundle atomically
    so an interrupted promotion cannot leave production without a model.

    Returns True when the model was promoted, False when promotion was refused.
    Callers must check the result — a refused promotion leaves the previous model
    serving, which is safe but means the challenger did not go live.
    """

    experiments_dir = Path(CONFIG["paths"]["experiments_dir"])
    target = Path(CONFIG["paths"]["production_model"]).parent

    source = experiments_dir / version

    if not source.exists():
        raise ValueError(f"Model version {version} does not exist.")

    new_metadata_path = source / "metadata.json"

    if not new_metadata_path.exists():
        raise ValueError("metadata.json missing for experiment.")

    new_metadata = load_metadata(new_metadata_path)

    parent_model = None
    existing_metadata_path = target / "metadata.json"

    # Schema safety check — serving depends on an exact, ordered feature match.
    if existing_metadata_path.exists():
        prod_metadata = load_metadata(existing_metadata_path)

        parent_model = prod_metadata.get("model_version")

        if not schemas_match(prod_metadata, new_metadata):
            logger.error(
                "Feature schema mismatch — promotion of %s blocked. "
                "Production has %d features, challenger has %d.",
                version,
                len(prod_metadata.get("feature_schema", [])),
                len(new_metadata.get("feature_schema", [])),
            )
            return False

    swap_model_bundle(source, target)

    logger.info("Model %s promoted to production.", version)

    # If this experiment was logged to the MLflow Model Registry, promote its stage too.
    try:
        mlflow_uri = new_metadata.get("mlflow_model_uri")
        if (
            mlflow_uri
            and isinstance(mlflow_uri, str)
            and mlflow_uri.startswith("models:/")
        ):
            import mlflow

            client = mlflow.tracking.MlflowClient()
            _, rest = mlflow_uri.split("models:/", 1)
            name, ver = rest.split("/", 1)
            client.transition_model_version_stage(
                name=name,
                version=ver,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("MLflow registry promoted to Production stage.")
    except Exception:
        logger.exception("MLflow stage promotion failed (non-blocking)")

    record_lineage(
        model_version=version,
        metrics=new_metadata.get("metrics", {}),
        dataset_used=new_metadata.get("dataset", "unknown"),
        trigger="drift_retraining",
        parent_model=parent_model,
    )

    return True
