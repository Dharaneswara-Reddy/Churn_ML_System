"""
Feature Builder

Single source of truth for feature preparation.
Used by BOTH training and inference pipelines.
"""

import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["training"])

DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "CLTV",
]


TARGET_COLUMN = "Churn Value"


def build_features(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """
    Prepare model-ready features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe
    training : bool
        Indicates training mode (kept for future use)
    """

    df = df.copy()

    # Single place where the raw "Total Charges" quirk (blank strings for new
    # customers) is resolved. Training used to repeat this in two further places,
    # and every copy silently mapped an unparseable balance to 0.0 — a large,
    # invisible feature corruption. Coerced rows are counted so the loss is at
    # least observable.
    numeric_charges = pd.to_numeric(df["Total Charges"], errors="coerce")
    coerced = int(numeric_charges.isna().sum())
    if coerced:
        logger.warning(
            "Coerced %d unparseable 'Total Charges' value(s) to 0.0", coerced
        )
    df["Total Charges"] = numeric_charges.fillna(0)

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    df = df.drop(
        columns=[c for c in DROP_COLUMNS if c in df.columns],
        errors="ignore",
    )

    return df
