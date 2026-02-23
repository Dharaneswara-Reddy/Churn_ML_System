"""
Data Validation Step

Ensures dataset satisfies schema and training requirements.
"""

import pandas as pd

from churn_system.schema import (
    validate_training_data,
)
from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["training"])


def run_data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate training dataset before feature engineering.
    """

    logger.info("Running data validation step")

    validate_training_data(df)

    logger.info("Data validation successful")

    return df