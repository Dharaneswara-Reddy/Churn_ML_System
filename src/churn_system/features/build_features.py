"""
Feature Builder

Single source of truth for feature preparation.
Used by BOTH training and inference pipelines.
"""

import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["training"])

# Columns that must never reach the model because they leak the target or are
# post-outcome information only available after churn has happened.
#
#   Churn Label  — the target restated as a string
#   Churn Score  — IBM's own churn propensity score; keeping it produces a
#                  ~1.0 ROC-AUC and a completely fraudulent model
#   Churn Reason — populated only for customers who already churned
#   CLTV         — lifetime value, computed with outcome knowledge
#   CustomerID   — a row identifier; memorisable, never generalisable
#   Count        — a constant 1 in this dataset
LEAKAGE_COLUMNS = [
    "CustomerID",
    "Count",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "CLTV",
]

# Geographic identifiers. Excluded deliberately, not incidentally:
#
#   * They accounted for 4,428 of 4,478 encoded columns (98.9%) on 5,634 training
#     rows — roughly one column per row, which is memorisation, not learning.
#   * "Zip Code" and "Lat Long" are 1:1 duplicates of each other; "Country" and
#     "State" are constant in this dataset and contribute permanently-zero columns.
#   * They crippled RandomForest: with max_features="sqrt" over 4,478 columns,
#     nearly every sampled feature was a near-empty ZIP indicator. Removing them
#     moved its recall from 0.016 to 0.543.
#   * Their category values are embedded as plaintext strings inside model.pkl,
#     which is how customer geography reached a public git repository, and they
#     are what /explain disclosed 1,650 coordinate pairs from.
#   * They are stripped by PII redaction before storage, so they can never be
#     drift-monitored and make labelled feedback rows unusable for retraining.
#
# Removing them here means they never enter the pipeline at all — they are not
# fitted, not encoded, and not present in feature_schema.
GEOGRAPHIC_COLUMNS = [
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]

DROP_COLUMNS = LEAKAGE_COLUMNS + GEOGRAPHIC_COLUMNS


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
