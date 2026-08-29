"""
Data Ingestion Step

Responsible for loading the training dataset from the configured source.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["training"])


RAW_DATA_PATH_ENV = "CHURN_RAW_DATA_PATH"


def resolve_training_data_path() -> Path:
    """
    Choose the dataset to train on.

    Order of precedence:

    1. **An explicit ``CHURN_RAW_DATA_PATH``.** Naming a dataset by hand is an
       instruction, not a hint, and it must win. It previously did not: the
       retraining-dataset preference below silently outranked it, so a CI smoke run
       configured to train on a synthetic CSV trained on ``data/retraining_dataset.csv``
       instead wherever that file happened to exist — real customer data, on a run
       whose whole purpose was to avoid it. The override was accepted, logged, and
       ignored.
    2. **The retraining dataset**, when the lifecycle has produced one. Without this
       preference the retraining dataset was written on every drift cycle and never
       read, so "retraining on fresh data" always re-fit the original static CSV.
    3. **The configured raw dataset.**
    """
    retraining_path = Path(CONFIG["paths"]["retraining_data"])
    raw_path = Path(CONFIG["paths"]["raw_data"])

    if os.environ.get(RAW_DATA_PATH_ENV):
        logger.info(
            "Using explicitly configured dataset from %s: %s (retraining dataset "
            "at %s bypassed)",
            RAW_DATA_PATH_ENV,
            raw_path,
            retraining_path,
        )
        return raw_path

    if retraining_path.exists():
        logger.info("Using retraining dataset: %s", retraining_path)
        return retraining_path

    logger.info("No retraining dataset present; using raw dataset: %s", raw_path)
    return raw_path


def load_training_data() -> tuple[pd.DataFrame, Path]:
    """
    Load the dataset used for model training.

    Returns
    -------
    tuple[pd.DataFrame, Path]
        Loaded dataframe and source data path.
    """
    data_path = resolve_training_data_path()

    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    logger.info("Loading training data from %s", data_path)

    df = pd.read_csv(data_path)

    logger.info("Dataset loaded | rows = %d | cols = %d", len(df), len(df.columns))

    return df, data_path
