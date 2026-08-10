"""
LifeCycle Scheduler

Runs the ML lifecycle pipeline periodically.
Simulates production automation.
"""

import time
from datetime import datetime, timezone

from churn_system.config.config import CONFIG
from churn_system.lifecycle.orchestrator import run_lifecycle
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])

CHECK_INTERVAL = CONFIG["scheduler"]["interval_seconds"]

def start_scheduler():
    """
    Continously run lifecycle checks at fixed intervals.
    """

    logger.info("Lifecycle scheduler started.")

    while True:
        logger.info("Running lifecycle check at %s UTC", datetime.now(timezone.utc).isoformat())

        try:
            run_lifecycle()
        except Exception:
            logger.exception("Lifecycle execution failed")

        logger.info("Sleeping for %d seconds...", CHECK_INTERVAL)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    start_scheduler()
