"""
ML LifeCycle Orchestrator

Runs monitoring pipeline and decides whether model retraining should be executed.
"""

import json
from pathlib import Path

from requests import get

from churn_system.lifecycle.model_compare import compare_models
from churn_system.monitoring.model_health import evaluate_model_health
from churn_system.training.train import main as train_model
from churn_system.new_data.retraining_data import build_retraining_dataset
from churn_system.lifecycle.promote import promote_model
from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__,CONFIG["logging"]["lifecycle"])


HEALTH_FILE = Path("models/monitoring/health_report.json")

def run_lifecycle():
    """
    Execute Monitoring -> decision -> retraining workflow
    """
    
    print("\n --- Evaluation Started ---")
    
    evaluate_model_health()
    
    if not HEALTH_FILE.exists():
        print("Health report missing. Aborting lifecycle.")
        return 
    
    with open(HEALTH_FILE, "r") as f:
        report = json.load(f)
        
    retrain_needed = report.get("retraining_recommended", False)
    
    if retrain_needed:
        print("\n Drift Identified - preparing retraining data.")
        build_retraining_dataset()
        print("Started retraining...")
        train_model()
        print("Evaluating challenger model...")
        
        if compare_models():
            print("Challenger wins - promoting model.")
            latest_version = sorted(
                Path("models/experiments").glob("churn_model_*")
            )[-1].name
            
            promote_model(latest_version)
        
    else:
        print("Challenger Rejected. Keep current production model.")
        print("\n Model healthy, No retraining triggered.")
        
    print("\n --- Evaluation is Completed ---")
    

if __name__ == "__main__":
    run_lifecycle()