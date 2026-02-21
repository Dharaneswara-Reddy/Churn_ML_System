"""
Champion vs Challenger Comparison.

Compares the current production (champion) model with the
latest experiment (challenger) model using evaluation metrics
and feature schema compatibility checks.
"""

import json
from pathlib import Path

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG
from churn_system.lifecycle.schema_compare import compare_feature_schemas


PRODUCTION_MODEL = Path("models/production/current/metadata.json")
EXPERIMENTS_DIR = Path("models/experiments")

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])


def get_latest_experiment():
    """
    Return latest experiment directory based on naming order.
    """
    versions = sorted(EXPERIMENTS_DIR.glob("churn_model_*"))
    return versions[-1] if versions else None


def load_metrics(path: Path):
    """
    Load evaluation metrics stored inside metadata.json.
    """
    with open(path, "r") as f:
        data = json.load(f)

    return data.get("metrics", {})


def compare_models():
    """
    Compare Production model with latest retrained model.

    Decision logic:
        1. Ensure schema compatibility.
        2. Compare ROC-AUC performance.
        3. Promote only if challenger is better AND safe.
    """

    latest = get_latest_experiment()

    if latest is None:
        logger.warning("No experiment models found.")
        return False

    challenger_meta = latest / "metadata.json"
    challenger_metrics = load_metrics(challenger_meta)

    # First deployment case
    if not PRODUCTION_MODEL.exists():
        logger.info("No production model found. Auto-promoting first model.")
        return True

    champion_metrics = load_metrics(PRODUCTION_MODEL)


    try:
        schema_report = compare_feature_schemas(
            PRODUCTION_MODEL,
            challenger_meta
        )

        logger.info(f"Schema comparison result: {schema_report}")

        # BLOCK promotion if breaking change detected
        if schema_report["removed_features"]:
            logger.warning(
                "Breaking schema change detected. "
                "Promotion blocked due to removed features."
            )
            return False

    except Exception as e:
        logger.error(f"Schema comparison failed: {e}")
        return False


    champion_auc = champion_metrics.get("roc_auc", 0)
    challenger_auc = challenger_metrics.get("roc_auc", 0)

    logger.info("--- Champion vs Challenger ---")
    logger.info(f"Champion ROC-AUC: {champion_auc}")
    logger.info(f"Challenger ROC-AUC: {challenger_auc}")

    if challenger_auc > champion_auc:
        logger.info("Challenger model wins.")
        return True

    logger.info("Champion model retained.")
    return False