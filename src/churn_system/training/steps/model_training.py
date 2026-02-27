"""
Model Training Step

Trains multiple candidate models and returns the best one.
Also records experiment results.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["training"])


def build_preprocessor(X):
    """
    Build preprocessing pipeline shared across models.
    """

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor


def train_candidate_models(X_train, y_train):
    """
    Train multiple candidate models.
    """

    preprocessor = build_preprocessor(X_train)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42
        ),
    }

    trained_models = {}

    for name, model in candidates.items():
        logger.info(f"Training candidate model: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

    return trained_models