"""
Model Evaluation Step

Evaluates all candidate models and selects the best one.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["training"])


def evaluate_candidates(models, X_test, y_test):
    """
    Evaluate all models and return winner + experiment report.
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
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "pr_auc": float(average_precision_score(y_test, probs)),
        }

        logger.info(f"{name} ROC-AUC = {metrics['roc_auc']:.4f}")

        results[name] = metrics

        # winner selection rule
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model
            best_name = name

    experiment_report = {
        "candidates": results,
        "winner": best_name,
    }

    logger.info(f"Winner selected: {best_name}")

    return best_model, experiment_report, results[best_name]