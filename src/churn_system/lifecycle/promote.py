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


def schema_difference(
    prod_meta: dict[str, Any], new_meta: dict[str, Any]
) -> dict[str, list[str]]:
    """Describe a schema change in the terms an operator has to reason about."""
    prod_schema = list(prod_meta.get("feature_schema", []))
    new_schema = list(new_meta.get("feature_schema", []))

    removed = [c for c in prod_schema if c not in new_schema]
    added = [c for c in new_schema if c not in prod_schema]
    # Same members, different order. Serving reindexes columns into training
    # order, so a pure reorder is still a real change worth naming.
    reordered = (
        []
        if (removed or added)
        else [a for a, b in zip(prod_schema, new_schema, strict=False) if a != b]
    )

    return {"removed": removed, "added": added, "reordered": reordered}


def promote_model(version: str, *, allow_schema_change: bool = False) -> bool:
    """
    Promote a trained model version to production.

    Ensures schema compatibility before promotion, and swaps the bundle atomically
    so an interrupted promotion cannot leave production without a model.

    Returns True when the model was promoted, False when promotion was refused.
    Callers must check the result — a refused promotion leaves the previous model
    serving, which is safe but means the challenger did not go live.

    Deliberate schema changes
    -------------------------
    The schema gate exists because the API request model is generated from the
    champion's ``feature_schema``: promoting a model with different features
    silently rewrites the public API contract. Blocking that by default is right,
    because the common cause is an *accident* — a feature-engineering edit that
    nobody intended to ship as an API change.

    But some schema changes are the entire point of the retrain (removing the
    geographic features, for instance). ``allow_schema_change=True`` is the
    explicit, logged opt-in for those. It is deliberately keyword-only and
    deliberately never passed by the automated lifecycle in
    ``lifecycle/orchestrator.py``: a scheduler must not be able to change the API
    contract on its own, because there is no human in that loop to notice.

    Callers that pass it are responsible for the client-facing side of the change.
    ``api/schema_generator`` keeps removed fields accepted-and-ignored, so removals
    do not break existing clients; *additions* are genuinely breaking, since the
    new field becomes required.
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
            difference = schema_difference(prod_metadata, new_metadata)

            if not allow_schema_change:
                logger.error(
                    "Feature schema mismatch — promotion of %s blocked. "
                    "Production has %d features, challenger has %d "
                    "(removed=%s, added=%s). Re-run with allow_schema_change=True "
                    "if this API contract change is intended.",
                    version,
                    len(prod_metadata.get("feature_schema", [])),
                    len(new_metadata.get("feature_schema", [])),
                    difference["removed"],
                    difference["added"],
                )
                return False

            # Loud on purpose: this line is the audit record that a human chose to
            # change the API contract, and which fields moved.
            logger.warning(
                "Promoting %s with an APPROVED feature-schema change. "
                "removed=%s added=%s. Removed fields remain accepted-and-ignored by "
                "the request schema; added fields become REQUIRED and will break "
                "clients that do not send them.",
                version,
                difference["removed"],
                difference["added"],
            )
            if difference["added"]:
                logger.warning(
                    "This promotion ADDS required request fields (%s) — existing "
                    "clients will receive 422 until they are updated.",
                    difference["added"],
                )

    swap_model_bundle(source, target, sign=True)

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
