"""
Champion vs Challenger Comparison.
"""

import json
from pathlib import Path
from sys import version
from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

PRODUCTION_MODEL = Path("models/production/current/metadata.json")
EXPERIMENTS_DIR = Path("models/experiments")

logger = get_logger(__name__,CONFIG["logging"]["lifecycle"])


def get_latest_experiment():
    versions = sorted(EXPERIMENTS_DIR.glob("churn_model_*"))
    return versions[-1] if versions else None

def load_metrics(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("metrics", {})

def compare_models():
    """
    Compare Production model with latest retrained model.
    """    
    
    latest = get_latest_experiment()
    
    if latest is None:
        print("No experiment models found.")
        return False
    
    challenger_metrics = load_metrics(latest / "metadata.json")
    
    if not PRODUCTION_MODEL.exists():
        print("No production model. Auto-promoting first model.")
        return True
    
    champion_metrics = load_metrics(PRODUCTION_MODEL)
    
    print("--- Champion vs Challenger ---")
    
    print("Champion ROC-AUC", champion_metrics.get("roc_auc"))
    print("challenger ROC-AUC", challenger_metrics.get("roc_auc"))
    
    return challenger_metrics.get("roc_auc",0) > champion_metrics.get("roc_auc", 0)