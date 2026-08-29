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
from sklearn.model_selection import train_test_split

from churn_system.artifacts import sign_model_bundle
from churn_system.config.config import CONFIG
from churn_system.features.build_features import GEOGRAPHIC_COLUMNS, LEAKAGE_COLUMNS
from churn_system.logging.logger import get_logger
from churn_system.mlflow_utils import configure_mlflow, log_artifact, log_sklearn_model
from churn_system.schema import TARGET_COLUMN
from churn_system.training.feature_types import infer_feature_types
from churn_system.training.steps.calibration_check import measure_calibration

# Pipeline steps
from churn_system.training.steps.data_ingestion import load_training_data
from churn_system.training.steps.data_validation import run_data_validation
from churn_system.training.steps.feature_engineering import run_feature_engineering
from churn_system.training.steps.model_evaluation import (
    compute_metrics,
    evaluate_candidates,
    select_threshold,
    selection_metric,
)
from churn_system.training.steps.model_training import train_candidate_models

GLOBAL_SEED = 42
# Fraction of the dataset held out as a final, untouched test set.
TEST_SIZE = 0.2
# Fraction of the TRAINING split reserved for threshold selection and calibration.
VALIDATION_SIZE = 0.25
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
    # Stratified holdout split
    # ----------------------------
    # The previous implementation sorted by "Tenure Months" and cut at the 80th
    # percentile, describing itself as "time-aware". It was not:
    #
    #   * Tenure Months is a per-customer DURATION observed at a single snapshot,
    #     not a calendar date. Every row is measured at the same instant, so
    #     ordering by it does not order the data in time.
    #   * It orders rows by accumulated survival, which is a monotone proxy for
    #     NOT having churned — i.e. it partitions the data on the label. Measured:
    #     train churn 31.5% vs test churn 6.6%, a 4.8x difference, of which 44.7%
    #     is pure compositional selection (long-tenure customers are overwhelmingly
    #     on two-year contracts).
    #   * The model never saw tenure > 60 during fit, and gradient boosting cannot
    #     extrapolate, so every test row fell on the same side of every tenure split.
    #   * PR-AUC's floor IS the base rate, so comparing PR-AUC across retrains whose
    #     folds have different base rates compares nothing — the promotion gate could
    #     promote or reject on a base-rate change alone.
    #
    # Tenure Months remains a model feature (it is legitimately predictive); it is
    # simply no longer the split axis. Stratifying on the target keeps the holdout
    # representative of the population actually being scored.
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COLUMN],
        random_state=GLOBAL_SEED,
        shuffle=True,
    )

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    logger.info(
        "Stratified split | train=%d (churn %.4f) | test=%d (churn %.4f) | "
        "population churn %.4f",
        len(train_df),
        y_train.mean(),
        len(test_df),
        y_test.mean(),
        df[TARGET_COLUMN].mean(),
    )

    log_target_distribution(y_train)

    # ----------------------------
    # Validation split (carved from TRAIN, never from the holdout)
    # ----------------------------
    # The operating threshold and any calibration must be chosen on data the final
    # metrics are not computed on. Tuning either on the test set makes the reported
    # precision/recall optimistic by construction.
    fit_df, validation_df = train_test_split(
        train_df,
        test_size=VALIDATION_SIZE,
        stratify=train_df[TARGET_COLUMN],
        random_state=GLOBAL_SEED,
        shuffle=True,
    )
    y_fit = fit_df[TARGET_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]

    # ----------------------------
    # Feature engineering
    # ----------------------------
    X_fit = run_feature_engineering(fit_df)
    X_validation = run_feature_engineering(validation_df)
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

    candidate_models = train_candidate_models(X_fit, y_fit)

    # ----------------------------
    # Threshold selection (validation data only)
    # ----------------------------
    provisional_metric = selection_metric()
    provisional_winner = max(
        sorted(candidate_models),
        key=lambda name: compute_metrics(
            y_validation,
            candidate_models[name].predict_proba(X_validation)[:, 1],
            0.5,
        )[provisional_metric],
    )
    validation_probabilities = (
        candidate_models[provisional_winner].predict_proba(X_validation)[:, 1]
    )
    threshold_report = select_threshold(y_validation, validation_probabilities)
    operating_threshold = threshold_report["threshold"]

    # Calibration measured on validation, before any correction is considered.
    calibration_report = measure_calibration(y_validation, validation_probabilities)
    logger.info(
        "Calibration (validation) | mean_predicted=%.4f actual=%.4f ratio=%.3f "
        "brier=%.4f",
        calibration_report["mean_predicted_probability"],
        calibration_report["actual_positive_rate"],
        calibration_report["calibration_ratio"],
        calibration_report["brier_score"],
    )

    # ----------------------------
    # Refit on the full training set, then evaluate on the untouched holdout
    # ----------------------------
    logger.info("Refitting candidates on the full training split...")
    candidate_models = train_candidate_models(X_train, y_train)

    logger.info("Evaluating candidate models on the holdout...")

    pipeline, experiment_report, metrics = evaluate_candidates(
        candidate_models,
        X_test,
        y_test,
        threshold=operating_threshold,
    )
    experiment_report["threshold_selection"] = threshold_report
    experiment_report["calibration_validation"] = calibration_report

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
        "feature_schema": feature_schema,
        "feature_types": feature_types,
        "feature_count": len(feature_schema),
        "metrics": metrics,
        "dataset": str(data_path),
        # The threshold travels with the model it was tuned for. Serving reads it
        # from here rather than from a global config constant, so a model tuned at
        # 0.28 is never served at someone else's 0.5.
        "operating_threshold": operating_threshold,
        "threshold_selection": threshold_report,
        "calibration_validation": calibration_report,
        "split_strategy": "stratified holdout (stratify=Churn Value)",
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "global_seed": GLOBAL_SEED,
        "excluded_features": {
            "leakage": LEAKAGE_COLUMNS,
            "geographic": GEOGRAPHIC_COLUMNS,
        },
    }

    metadata_path = model_dir / "metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved.")

    # ----------------------------
    # Sign the bundle at creation
    # ----------------------------
    # Signing here, rather than only at promotion, closes a gap in the chain of
    # custody. Promotion used to sign whatever bundle it found in
    # models/experiments/ without verifying it first — so anyone able to write to
    # that directory could tamper with a model.pkl and have promotion bless it,
    # after which the API would verify the signature happily and unpickle it.
    #
    # The signature is computed over model.pkl and metadata.json together, so it
    # is written last: both files must already be final.
    signature_path = sign_model_bundle(model_dir)
    logger.info("Bundle signed at %s", signature_path)

    # ----------------------------
    # MLflow tracking + registry
    # ----------------------------
    if mlflow_cfg.get("enabled", True):
        import mlflow

        with mlflow.start_run(run_name=f"churn_model_{MODEL_VERSION}"):
            mlflow.log_params(
                {
                    "model_type": winner_name,
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
