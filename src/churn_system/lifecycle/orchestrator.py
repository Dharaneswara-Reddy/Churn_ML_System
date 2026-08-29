"""
ML LifeCycle Orchestrator

Runs monitoring pipeline and decides whether model retraining should be executed.
"""

from __future__ import annotations

import json
from pathlib import Path

from churn_system.config.config import CONFIG
from churn_system.lifecycle.model_compare import compare_models
from churn_system.lifecycle.promote import promote_model
from churn_system.lifecycle.rollback import rollback_if_needed
from churn_system.lifecycle.serving_reload import notify_serving_reload
from churn_system.logging.logger import get_logger
from churn_system.monitoring.model_health import evaluate_model_health
from churn_system.new_data.retraining_data import build_retraining_dataset
from churn_system.training.train import main as train_model

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])


def _health_file() -> Path:
    return Path(CONFIG["paths"]["monitoring_dir"]) / "health_report.json"


def run_lifecycle() -> dict[str, bool]:
    """
    Execute the monitoring -> decision -> retraining workflow.

    Returns a summary of what happened, so a scheduler or test can assert on the
    outcome instead of parsing logs.
    """
    logger.info("--- Lifecycle evaluation started ---")

    outcome = {"retrained": False, "promoted": False, "rolled_back": False}

    evaluate_model_health()

    health_file = _health_file()
    if not health_file.exists():
        logger.error("Health report missing at %s. Aborting lifecycle.", health_file)
        return outcome

    with open(health_file, encoding="utf-8") as f:
        report = json.load(f)

    if not report.get("retraining_recommended", False):
        logger.info("Model healthy. No retraining triggered.")
        # Still consider rollback: the running model may be unhealthy for reasons
        # unrelated to this cycle's drift verdict.
        outcome["rolled_back"] = rollback_if_needed()
        if outcome["rolled_back"]:
            # Rollback rewrites bytes on disk exactly like promotion does, so the
            # serving layer needs the same notification — otherwise the recovery
            # path silently leaves every replica on the model it rolled back from.
            notify_serving_reload()
        logger.info("--- Lifecycle evaluation completed ---")
        return outcome

    logger.info("Drift identified — preparing retraining data.")
    build_retraining_dataset()

    logger.info("Starting retraining...")
    trained_version = train_model()
    outcome["retrained"] = True

    logger.info("Evaluating challenger model %s...", trained_version)

    if compare_models():
        # Promote the version we just trained rather than re-globbing for the newest
        # directory — another training process could have produced a newer one.
        outcome["promoted"] = promote_model(trained_version)

        if outcome["promoted"]:
            logger.info("Challenger promoted: %s", trained_version)
            notify_serving_reload()
        else:
            logger.error(
                "Promotion of %s was refused (schema mismatch). "
                "Production still serves the previous model.",
                trained_version,
            )
    else:
        logger.info("Challenger rejected. Keeping current production model.")

    if outcome["promoted"]:
        # Do not roll back a model that was promoted seconds ago. The health report
        # still carries the drift verdict that triggered *this* cycle, so feeding it
        # to the rollback check would revert every successful promotion immediately.
        logger.info("Skipping rollback check — a new model was promoted this cycle.")
    else:
        outcome["rolled_back"] = rollback_if_needed()
        if outcome["rolled_back"]:
            notify_serving_reload()

    logger.info("--- Lifecycle evaluation completed ---")
    return outcome


if __name__ == "__main__":
    run_lifecycle()
