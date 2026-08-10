"""
Automatic Rollback System.

Reverts production model if current model is marked unhealthy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from churn_system.artifacts import swap_model_bundle
from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])


def _lineage_path() -> Path:
    return Path(CONFIG["paths"]["lineage_path"])


def _health_path() -> Path:
    return Path(CONFIG["paths"]["monitoring_dir"]) / "health_report.json"


def _previous_version(lineage: list[dict[str, Any]]) -> str | None:
    """
    Return the most recent lineage entry that differs from the current one.

    Lineage records every promotion, and repeated promotions of the same version
    are common, so ``lineage[-2]`` is frequently the version already in production.
    Rolling back to that is a no-op dressed up as a recovery.
    """
    current = lineage[-1].get("model_version")
    for entry in reversed(lineage[:-1]):
        version = entry.get("model_version")
        if version and version != current:
            return version
    return None


def rollback_if_needed() -> bool:
    """
    Roll back the production model if the health check fails.

    Returns True when a rollback was performed.
    """
    health_path = _health_path()
    lineage_path = _lineage_path()

    if not health_path.exists():
        logger.info("No health report found. Skipping rollback.")
        return False

    with open(health_path, encoding="utf-8") as f:
        health = json.load(f)

    if not health.get("retraining_recommended", False):
        logger.info("Model healthy. No rollback required.")
        return False

    if not lineage_path.exists():
        logger.error("No lineage history available.")
        return False

    with open(lineage_path, encoding="utf-8") as f:
        lineage = json.load(f)

    if len(lineage) < 2:
        logger.error("No previous model available for rollback.")
        return False

    previous_model = _previous_version(lineage)

    if previous_model is None:
        logger.error(
            "Lineage contains only one distinct model version; nothing to roll back to."
        )
        return False

    experiments_dir = Path(CONFIG["paths"]["experiments_dir"])
    production_dir = Path(CONFIG["paths"]["production_model"]).parent
    source = experiments_dir / previous_model

    if not source.exists():
        logger.error("Previous model folder missing: %s", source)
        return False

    swap_model_bundle(source, production_dir)

    logger.warning("Rollback completed -> restored %s", previous_model)
    return True
