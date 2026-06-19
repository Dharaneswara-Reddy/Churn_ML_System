"""
Model Evaluation Step

Evaluates all candidate models and selects the best one.
"""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["training"])

# Honour the configurable selection metric from settings.yaml
SELECTION_METRIC = CONFIG.get("training", {}).get("selection_metric", "roc_auc")


def evaluate_candidates(models, X_test, y_test):
    """
    Evaluate all models and return winner + experiment report.

    The winner is selected by the metric defined in
    ``CONFIG["training"]["selection_metric"]`` (default: ``roc_auc``).
    """

    results = {}
    best_model = None
    best_score = -1
    best_name = None

    for name, model in models.items():

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(
                precision_score(y_test, preds, zero_division=0)
            ),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1_score": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "pr_auc": float(average_precision_score(y_test, probs)),
        }

        logger.info(f"{name} {SELECTION_METRIC} = {metrics[SELECTION_METRIC]:.4f}")

        results[name] = metrics

        # winner selection rule — driven by CONFIG
        if metrics[SELECTION_METRIC] > best_score:
            best_score = metrics[SELECTION_METRIC]
            best_model = model
            best_name = name

    experiment_report = {
        "candidates": results,
        "winner": best_name,
        "selection_metric": SELECTION_METRIC,
    }

    logger.info(f"Winner selected: {best_name} ({SELECTION_METRIC}={best_score:.4f})")

    return best_model, experiment_report, results[best_name]
