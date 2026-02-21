"""
Training Pipeline

Responsible for orchestrating model training workflow.
Does not implement ML Logic - delegates to training module.
"""

from venv import logger
from churn_system.training.train import main as train_model
from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG


logger = get_logger(__name__, CONFIG["logging"]["training"])


def run_training_pipeline():
    """
    Execute end-to-end training workflow.
    """
    
    logger.info("--- Training Pipeline Started ---")
    
    try:
        train_model()
        logger.info("Training completed successfully.")
        
    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        raise
    
    logger.info("--- Training Pipeline Finished ---")
    
if __name__ == "__main__":
    run_training_pipeline()
    