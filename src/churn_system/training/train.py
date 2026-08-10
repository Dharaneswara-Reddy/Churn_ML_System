"""
Training Orchestrator

Coordinates the full ML training workflow:
Data → Validation → Feature Engineering → Training → Evaluation → Artifact Saving
"""

import json
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger
from churn_system.mlflow_utils import configure_mlflow, log_artifact, log_sklearn_model
from churn_system.schema import TARGET_COLUMN
from churn_system.training.feature_types import infer_feature_types

# Pipeline steps
from churn_system.training.steps.data_ingestion import load_training_data
from churn_system.training.steps.data_validation import run_data_validation
from churn_system.training.steps.feature_engineering import run_feature_engineering
from churn_system.training.steps.model_evaluation import evaluate_candidates
from churn_system.training.steps.model_training import train_candidate_models

GLOBAL_SEED = 42
logger = get_logger(__name__, CONFIG["logging"]["training"])


def new_model_version() -> str:
    """
    Build a fresh version stamp for a training run.

    Computed per call, never at import: the scheduler imports ``main`` once and then
    calls it every cycle, so an import-time stamp would make every retrain overwrite
    the same experiment directory and destroy the run history the lifecycle needs to
    compare and roll back.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def log_target_distribution(y):
    values, counts = np.unique(y, return_counts=True)
    dist = dict(zip(values, counts, strict=True))
    logger.info("Target distribution: %s", dist)


def summarize_feature(name, train_series, test_series):
    logger.info(
        "%s | Train(mean=%.2f, std=%.2f) | Test(mean=%.2f, std=%.2f)",
        name,
        train_series.mean(),
        train_series.std(),
        test_series.mean(),
        test_series.std(),
    )


def main() -> str:
    """
    Run the full training pipeline.

    Returns the experiment directory name (``churn_model_<version>``) so callers can
    act on the run they just produced instead of re-globbing for the newest
    directory, which is racy when more than one training process exists.
    """
    # ----------------------------
    # Reproducibility: global seeds
    # ----------------------------
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    MODEL_VERSION = new_model_version()

    logger.info("===== Training Pipeline Started =====")
    mlflow_cfg = configure_mlflow()

    # ----------------------------
    # Data ingestion
    # ----------------------------
    df, data_path = load_training_data()

    logger.info("Training dataset used: %s", data_path)
    logger.info("Training samples: %d", len(df))

    # ----------------------------
    # Data validation
    # ----------------------------
    df = run_data_validation(df)

    # "Total Charges" is normalised once, in build_features — the single transform
    # shared by training and serving. Repeating it here risked the two paths
    # diverging, which is exactly the train/serve skew the shared builder exists
    # to prevent.

    # ----------------------------
    # Temporal split
    # ----------------------------
    df_sorted = df.sort_values("Tenure Months")

    split_index = int(0.8 * len(df_sorted))
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    log_target_distribution(y_train)

    # ----------------------------
    # Feature engineering
    # ----------------------------
    X_train = run_feature_engineering(train_df)
    X_test = run_feature_engineering(test_df)

    feature_schema = list(X_train.columns)
    feature_types = infer_feature_types(X_train)

    logger.info("Feature schema captured (%d features)", len(feature_schema))

    # ----------------------------
    # Feature statistics
    # ----------------------------
    summarize_feature(
        "Tenure Months",
        train_df["Tenure Months"],
        test_df["Tenure Months"],
    )

    summarize_feature(
        "Monthly Charges",
        train_df["Monthly Charges"],
        test_df["Monthly Charges"],
    )

    summarize_feature(
        "Total Charges",
        train_df["Total Charges"],
        test_df["Total Charges"],
    )

    logger.info(
        "Train tenure range: %s - %s",
        train_df["Tenure Months"].min(),
        train_df["Tenure Months"].max(),
    )

    logger.info(
        "Test tenure range: %s - %s",
        test_df["Tenure Months"].min(),
        test_df["Tenure Months"].max(),
    )

    # ----------------------------
    # Save training reference
    # ----------------------------
    reference_path = Path(CONFIG["paths"]["training_reference"])
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(reference_path, index=False)

    logger.info("Training reference data saved.")

    # ----------------------------
    # Train candidate models
    # ----------------------------
    logger.info("Training candidate models...")

    candidate_models = train_candidate_models(X_train, y_train)

    # ----------------------------
    # Evaluate candidates
    # ----------------------------
    logger.info("Evaluating candidate models...")

    pipeline, experiment_report, metrics = evaluate_candidates(
        candidate_models,
        X_test,
        y_test,
    )

    winner_name = experiment_report["winner"]

    logger.info("Champion model selected: %s", winner_name)

    # ----------------------------
    # Save artifacts
    # ----------------------------
    model_dir = (
        Path(CONFIG["paths"]["experiments_dir"])
        / f"churn_model_{MODEL_VERSION}"
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info("Model saved at %s", model_path)

    # ----------------------------
    # Save experiment report
    # ----------------------------
    report_path = model_dir / "experiment_report.json"

    with open(report_path, "w") as f:
        json.dump(experiment_report, f, indent=2)

    logger.info("Experiment report saved.")

    # ----------------------------
    # Save metadata
    # ----------------------------
    metadata = {
        "model_version": MODEL_VERSION,
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model_type": winner_name,
        "split_strategy": "time-aware (tenure-based)",
        "feature_schema": feature_schema,
        "feature_types": feature_types,
        "feature_count": len(feature_schema),
        "metrics": metrics,
        "dataset": str(data_path),
    }

    metadata_path = model_dir / "metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved.")

    # ----------------------------
    # MLflow tracking + registry
    # ----------------------------
    if mlflow_cfg.get("enabled", True):
        import mlflow

        with mlflow.start_run(run_name=f"churn_model_{MODEL_VERSION}"):
            mlflow.log_params(
                {
                    "model_type": winner_name,
                    "split_strategy": "time-aware (tenure-based)",
                    "feature_count": len(feature_schema),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tag("model_version", MODEL_VERSION)
            mlflow.set_tag("dataset_path", str(data_path))

            model_uri = log_sklearn_model(
                pipeline=pipeline,
                registered_model_name=mlflow_cfg["registered_model_name"],
                tags={"winner": winner_name},
            )
            mlflow.set_tag("mlflow_model_uri", model_uri)

            log_artifact(report_path)
            log_artifact(metadata_path)

            # Persist MLflow pointers into your metadata.json for downstream promotion
            try:
                run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
                metadata_update = dict(metadata)
                metadata_update["mlflow_run_id"] = run_id
                metadata_update["mlflow_model_uri"] = model_uri
                with open(metadata_path, "w") as f:
                    json.dump(metadata_update, f, indent=2)
            except Exception:
                logger.exception("Failed to update metadata with MLflow run info")
    else:
        logger.info("MLflow disabled by CHURN_MLFLOW_ENABLED")

    logger.info("===== Training Pipeline Completed =====")

    return model_dir.name


if __name__ == "__main__":
    main()
