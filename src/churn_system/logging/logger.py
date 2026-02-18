"""
Central logging configuration for the churn system.

Supports:
- Separate log files per subsystem
- Console + file logging
- Log rotation (prevents huge files)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, logfile: str = "system.log") -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Module name requesting logger.

    logfile : str
        Log file name (training.log, api.log, monitoring.log, etc.)

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers when modules reload
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_path = LOG_DIR / logfile

    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)


    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
