"""
Model Evaluation Step

Evaluates trained model and computes performance metrics.
"""

import numpy as np
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

def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained pipeline on test data.
    Returns metrics dictionary and prediction probabilites.
    """
    
    logger.info("Starting model evaluation step")
    probs = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)
    
    metrics = {
        "accuracy" : float(accuracy_score(y_test, preds)),
        "precision" : float(precision_score(y_test, preds)),
        "recall" : float(recall_score(y_test, preds)),
        "f1_score" : float(f1_score(y_test, preds)),
        "roc_auc" : float(roc_auc_score(y_test, probs)),
        "pr_auc" : float(average_precision_score(y_test, probs)),
    }
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"f1_score: {metrics['f1_score']:.4f}")
    logger.info(f"pr_auc: {metrics['pr_auc']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    
    for t in [0.3,0.5,0.7]:
        preds_t = (probs >= t).astype(int)
        logger.info(
            f"Threshold {t} |"
            f"Precision = {precision_score(y_test, preds_t):.4f} | "
            f"Recall = {recall_score(y_test, preds_t):.3f}"
        )
    logger.info("Model evaluation completed")
    return metrics, probs