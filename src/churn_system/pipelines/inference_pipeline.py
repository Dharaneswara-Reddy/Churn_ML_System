"""
Inference Pipeline

Handles prediction workflow used by API layer.
"""

from churn_system.inference.inference import run_inference
from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["api"])

def run_inference_pipeline(payload: dict):
    """
    Execute inference workflow.
    """
    
    logger.info("Inference pipeline started")
    
    try:
        result = run_inference(payload)
        logger.info("Inference pipeline completed")
        return result
    except Exception as e:
        logger.exception(f"Inference pipeline failed: {e}")
        raise
    