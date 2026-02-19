"""
Automatic Rollback System.

Reverts production model if current model is marked unhealthy.
"""

import json
import shutil
from pathlib import Path

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])

LINEAGE_PATH = Path("models/lineage/lineage.json")
HEALTH_PATH = Path("models/monitoring/health_report.json")


def rollback_if_needed():
    """
    Roll back production model if health check fails.
    """

    if not HEALTH_PATH.exists():
        logger.info("No health report found. Skipping rollback.")
        return

    with open(HEALTH_PATH, "r") as f:
        health = json.load(f)

    if not health.get("retraining_recommended", False):
        logger.info("Model healthy. No rollback required.")
        return

    if not LINEAGE_PATH.exists():
        logger.error("No lineage history available.")
        return

    with open(LINEAGE_PATH, "r") as f:
        lineage = json.load(f)

    if len(lineage) < 2:
        logger.error("No previous model available for rollback.")
        return

    previous_model = lineage[-2]["model_version"]

    experiments_dir = Path(CONFIG["paths"]["experiments_dir"])
    production_dir = Path(CONFIG["paths"]["production_model"]).parent / "current"

    source = experiments_dir / previous_model

    if not source.exists():
        logger.error("Previous model folder missing.")
        return

    shutil.rmtree(production_dir)
    shutil.copytree(source, production_dir)

    logger.warning(f"Rollback completed -> restored {previous_model}")
