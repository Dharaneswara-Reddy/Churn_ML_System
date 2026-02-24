"""
Feature Engineering Step

Builds model-ready features using shared feature builder.
"""

import pandas as pd

from churn_system.features.build_features import build_features
from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["training"])


def run_feature_engineering(df: pd.DataFrame):
    """
    Transform validated dataset into model features.
    """

    logger.info("Running feature engineering step")

    X = build_features(df, training=True)

    logger.info(f"Feature engineering completed | features={len(X.columns)}")

    return X