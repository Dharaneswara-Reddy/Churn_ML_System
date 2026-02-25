"""
Data Ingestion Step

Responsible for loading training dataset from configured source.
"""

import pandas as pd
from pathlib import Path

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["training"])


def load_training_data():
    """
    Load raw dataset used for model training.

    Returns
    -------
    tuple[pd.DataFrame, Path]
        Loaded dataframe and source data path.
    """

    data_path = Path(CONFIG["paths"]["raw_data"])

    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    logger.info(f"Loading training data from {data_path}")

    df = pd.read_csv(data_path)

    logger.info(
        f"Dataset loaded | rows = {len(df)} | cols = {len(df.columns)}"
    )


    return df, data_path