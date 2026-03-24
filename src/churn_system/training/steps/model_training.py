import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["training"])


def build_preprocessor(X):
    # Handle both legacy object dtype and pandas StringDtype ("string")
    # to avoid categorical values leaking into numeric scaler paths.
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns
    numerical_cols = X.select_dtypes(include=[np.number, "bool"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor


def get_model_registry():
    """
    Defines candidate models for competition.
    """

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            random_state=42,
        ),
    }


def train_candidate_models(X_train, y_train):
    """
    Train all registered candidates and return name -> fitted Pipeline.

    Each candidate gets its own preprocessor instance to avoid shared-state bugs
    across sklearn Pipeline fits.
    """
    registry = get_model_registry()
    fitted: dict[str, Pipeline] = {}

    for name, estimator in registry.items():
        logger.info(f"Training candidate model: {name}")
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline

    return fitted
