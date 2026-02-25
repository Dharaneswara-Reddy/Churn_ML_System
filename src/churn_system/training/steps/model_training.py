"""
Model Training Step

Creates preprocessing pipeline and trains ML model.
"""

import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["training"])


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train ML pipeline and return fitted model.
    """

    logger.info("Starting model training step")

    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    pipeline.fit(X_train, y_train)

    logger.info("Model training completed")

    return pipeline