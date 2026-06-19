"""
Central logging configuration for the churn system.

Supports:
- JSON-structured logs for production (machine-readable)
- Human-readable logs for development
- Separate log files per subsystem
- Console + file logging
- Log rotation (prevents huge files)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set CHURN_LOG_FORMAT=json for structured production logs
_JSON_MODE = os.environ.get("CHURN_LOG_FORMAT", "text").lower() == "json"


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        # Merge extra structured fields passed via `logger.info("msg", extra={...})`
        for key in ("model_id", "model_version", "request_id", "latency_ms",
                     "prediction", "confidence", "feature_hash", "path", "method"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, default=str)


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

    if _JSON_MODE:
        formatter: logging.Formatter = JSONFormatter()
    else:
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
