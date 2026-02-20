"""
Feature Builder

Single source of truth for feature preparation.
Used by BOTH training and inference pipelines.
"""

import pandas as pd


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

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"], errors="coerce"
    ).fillna(0)

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    df = df.drop(
        columns=[c for c in DROP_COLUMNS if c in df.columns],
        errors="ignore",
    )

    return df
