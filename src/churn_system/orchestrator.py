"""
ML LifeCycle Orchestrator

Runs monitoring pipeline and decides whether model retraining should be executed.
"""

import json
from pathlib import Path

from churn_system.model_health import evaluate_model_health
from churn_system.training import main as train_model
from churn_system.retraining_data import build_retraining_dataset

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
        print("Retraining Completed")
        
    else:
        print("\n Model healthy, No retraining triggered.")
        
    print("\n --- Evaluation is Completed ---")
    

if __name__ == "__main__":
    run_lifecycle()