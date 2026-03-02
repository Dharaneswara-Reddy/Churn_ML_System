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


def get_model_registry():
    """
    Defines candidate models for competition.
    """

    return {
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


def train_model(X_train, y_train, X_test, y_test):
    """
    Train multiple models and return the best performer.
    """

    preprocessor = build_preprocessor(X_train)
    models = get_model_registry()

    best_pipeline = None
    best_metrics = None
    best_score = -1
    best_name = None

    from churn_system.training.steps.model_evaluation import evaluate_model

    for name, model in models.items():

        logger.info(f"Training candidate model: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics, _ = evaluate_model(pipeline, X_test, y_test)

        score = metrics["roc_auc"]

        logger.info(f"{name} ROC-AUC = {score:.4f}")

        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_metrics = metrics
            best_name = name

    logger.info(f"Champion selected: {best_name}")

    return best_pipeline, best_metrics, best_name