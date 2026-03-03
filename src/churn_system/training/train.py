"""
Model Training Step

Trains candidate models but does NOT evaluate them.
Evaluation happens in the evaluation step.
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

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor


def train_model(X_train, y_train):
    """
    Train candidate models and return pipelines.
    """

    logger.info("Building preprocessing pipeline")

    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            random_state=42
        ),
    }

    trained_models = {}

    for name, model in models.items():

        logger.info(f"Training candidate model: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

    logger.info(f"Candidate models trained: {list(trained_models.keys())}")

    return trained_models