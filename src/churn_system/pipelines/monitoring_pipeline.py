"""
Monitoring Pipeline

Runs model monitoring checks:
- drift detection 
- health evaluation
"""

from churn_system.monitoring.model_health import evaluate_model_health
from churn_system.monitoring.prediction_monitor import generate_prediction_report
from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["monitoring"])

def run_monitoring_pipeline():
    logger.info("--- Monitoring Pipeline Started ---")
    
    try:
        evaluate_model_health()
        generate_prediction_report()
        logger.info("Monitoring checks completed.")
    except Exception as e:
        logger.exception(f"Monitoring pipeline failed: {e}")
        raise
    
    logger.info("--- Monitoring Pipeline Finished ---")
    
if __name__ == "__main__":
    run_monitoring_pipeline()