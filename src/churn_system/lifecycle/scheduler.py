"""
LifeCycle Scheduler

Runs the ML lifecycle pipeline periodically.
Simulates production automation.
"""

import time
from datetime import datetime,timezone

from churn_system.lifecycle.orchestrator import run_lifecycle
from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])

CHECK_INTERVAL = CONFIG["scheduler"]["interval_seconds"]

def start_scheduler():
    """
    Continously run lifecycle checks at fixed intervals.
    """
    
    logger.info("Lifecycle scheduler started.")
    
    while True:
        logger.info(f"Running lifecycle check at {datetime.now(timezone.utc).isoformat()} UTC")
        
        try:
            run_lifecycle()
        except Exception as e:
            logger.exception(f"Lifecycle execution failed: {e}")
            
        logger.info(f"Sleeping for {CHECK_INTERVAL} seconds...")
        time.sleep(CHECK_INTERVAL)
        
if __name__ == "__main__":
    start_scheduler()
