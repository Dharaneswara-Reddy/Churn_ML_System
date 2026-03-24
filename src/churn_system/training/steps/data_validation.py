"""
Data Validation Step

Ensures dataset satisfies schema and training requirements.
"""

from pathlib import Path

import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger
from churn_system.schema import (
    validate_training_data,
)
from churn_system.validation.validator import validate_dataframe

logger = get_logger(__name__, CONFIG["logging"]["training"])


def run_data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate training dataset before feature engineering.
    """

    logger.info("Running data validation step")

    # Clean known raw-data quirks before strict contract validation.
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"].replace(" ", pd.NA),
            errors="coerce",
        ).fillna(0.0)
    if "Churn Reason" in df.columns:
        df["Churn Reason"] = df["Churn Reason"].fillna("Unknown")

    validate_training_data(df)
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "validation"
        / "schemas"
        / "v1.yaml"
    )
    df = validate_dataframe(df, schema_path=schema_path)

    logger.info("Data validation successful")

    return df
