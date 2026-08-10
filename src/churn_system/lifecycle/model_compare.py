"""
Champion vs Challenger Comparison.

Compares the current production (champion) model with the
latest experiment (challenger) model using evaluation metrics
and feature schema compatibility checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from churn_system.artifacts import latest_experiment_dir, load_metadata
from churn_system.config.config import CONFIG
from churn_system.lifecycle.schema_compare import compare_feature_schemas
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])


def _production_metadata_path() -> Path:
    return Path(CONFIG["paths"]["production_model"]).parent / "metadata.json"


def _promotion_metric() -> str:
    return str(CONFIG.get("model_promotion", {}).get("metric", "pr_auc"))


def _min_improvement() -> float:
    return float(CONFIG.get("model_promotion", {}).get("min_improvement", 0.0))


def get_latest_experiment() -> Path | None:
    """Return the newest complete experiment bundle, or None."""
    return latest_experiment_dir()


def load_metrics(path: Path) -> dict[str, Any]:
    """Load evaluation metrics stored inside metadata.json."""
    return load_metadata(path).get("metrics", {})


def compare_models() -> bool:
    """
    Compare the production model with the latest retrained model.

    Decision logic:
        1. Ensure schema compatibility.
        2. Compare the configured promotion metric.
        3. Promote only if the challenger improves on it by at least
           ``model_promotion.min_improvement``.

    The metric and margin come from configuration rather than being hardcoded: the
    training pipeline selects its winner by ``training.selection_metric`` (PR-AUC by
    default, which suits this imbalanced target), so judging promotion by a
    different metric can reject the very model training just chose.
    """

    latest = get_latest_experiment()

    if latest is None:
        logger.warning("No experiment models found.")
        return False

    metric = _promotion_metric()
    min_improvement = _min_improvement()

    challenger_meta = latest / "metadata.json"
    challenger_metrics = load_metrics(challenger_meta)

    production_metadata = _production_metadata_path()

    # First deployment case
    if not production_metadata.exists():
        logger.info("No production model found. Auto-promoting first model.")
        return True

    champion_metrics = load_metrics(production_metadata)

    try:
        schema_report = compare_feature_schemas(production_metadata, challenger_meta)

        logger.info("Schema comparison result: %s", schema_report)

        # BLOCK promotion if breaking change detected
        if schema_report["removed_features"]:
            logger.warning(
                "Breaking schema change detected. "
                "Promotion blocked due to removed features."
            )
            return False

    except Exception:
        logger.exception("Schema comparison failed — refusing to promote")
        return False

    if metric not in champion_metrics or metric not in challenger_metrics:
        logger.error(
            "Promotion metric %r missing (champion=%s, challenger=%s). "
            "Refusing to promote on incomparable metrics.",
            metric,
            sorted(champion_metrics),
            sorted(challenger_metrics),
        )
        return False

    champion_score = float(champion_metrics[metric])
    challenger_score = float(challenger_metrics[metric])
    improvement = challenger_score - champion_score

    logger.info("--- Champion vs Challenger (%s) ---", metric)
    logger.info("Champion   %s: %.6f", metric, champion_score)
    logger.info("Challenger %s: %.6f", metric, challenger_score)
    logger.info("Improvement: %+.6f (required: %+.6f)", improvement, min_improvement)

    if improvement >= min_improvement and improvement > 0:
        logger.info("Challenger model wins.")
        return True

    logger.info("Champion model retained.")
    return False
