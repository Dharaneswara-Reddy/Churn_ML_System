"""
Concurrent Model Training

Trains candidate models in PARALLEL using ThreadPoolExecutor with
proper synchronization for shared result collection.

Why ThreadPoolExecutor (not ProcessPoolExecutor):
  - sklearn releases the GIL during C-extension computation (BLAS/LAPACK)
  - Threads share memory → no expensive serialization of DataFrames
  - Each candidate gets its own preprocessor to avoid shared-state bugs

Synchronization primitives used:
  - threading.Lock:  protects the shared `fitted` results dict
  - concurrent.futures.Future: provides thread-safe result collection
  - as_completed(): enables eager result processing as models finish
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            random_state=42,
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


def _train_single_candidate(name: str, estimator, X_train, y_train) -> tuple[str, Pipeline]:
    """
    Train a single candidate model. Designed to be submitted to a thread pool.

    Each invocation builds its own preprocessor instance to guarantee no
    shared mutable state between threads.
    """
    logger.info(f"[Thread {threading.current_thread().name}] Training candidate: {name}")
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    logger.info(f"[Thread {threading.current_thread().name}] Finished training: {name}")
    return name, pipeline


def train_candidate_models(X_train, y_train):
    """
    Train all registered candidates CONCURRENTLY and return name -> fitted Pipeline.

    Uses ThreadPoolExecutor for parallel training:
    - sklearn releases the GIL during BLAS/LAPACK operations, so threads
      achieve genuine parallelism on C-extension code paths.
    - A threading.Lock guards the shared results dictionary.
    - as_completed() processes results as soon as each model finishes.
    """
    registry = get_model_registry()
    fitted: dict[str, Pipeline] = {}
    results_lock = threading.Lock()

    max_workers = CONFIG.get("training", {}).get("max_workers", len(registry))

    logger.info(
        f"Starting concurrent training of {len(registry)} candidates "
        f"with max_workers={max_workers}"
    )

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="trainer") as executor:
        # Submit all training jobs
        future_to_name = {
            executor.submit(
                _train_single_candidate, name, estimator, X_train, y_train
            ): name
            for name, estimator in registry.items()
        }

        # Collect results as they complete (not in submission order)
        for future in as_completed(future_to_name):
            submitted_name = future_to_name[future]
            try:
                name, pipeline = future.result()
                with results_lock:
                    fitted[name] = pipeline
                logger.info(f"Candidate '{name}' training completed successfully")
            except Exception:
                logger.exception(f"Candidate '{submitted_name}' training FAILED")
                raise

    logger.info(f"All {len(fitted)} candidates trained successfully")
    return fitted
