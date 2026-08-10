"""
Data Ingestion Step

Responsible for loading the training dataset from the configured source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["training"])


def resolve_training_data_path() -> Path:
    """
    Choose the dataset to train on.

    Prefers the retraining dataset when the lifecycle has produced one, falling back
    to the raw dataset otherwise. Without this preference the retraining dataset was
    written on every drift cycle and never read, so "retraining on fresh data" always
    re-fit the original static CSV.
    """
    retraining_path = Path(CONFIG["paths"]["retraining_data"])
    raw_path = Path(CONFIG["paths"]["raw_data"])

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
