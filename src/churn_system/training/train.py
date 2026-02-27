"""
Training Orchestrator

Coordinates the full ML training workflow:
Data → Validation → Feature Engineering → Training → Evaluation → Artifact Saving
"""

import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

from churn_system.logging.logger import get_logger
from churn_system.schema import TARGET_COLUMN
from churn_system.config.config import CONFIG

# pipeline steps
from churn_system.training.steps.data_ingestion import load_training_data
from churn_system.training.steps.data_validation import run_data_validation
from churn_system.training.steps.feature_engineering import run_feature_engineering
from churn_system.training.steps.model_training import train_candidate_models
from churn_system.training.steps.model_evaluation import evaluate_candidates


MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = get_logger(__name__, CONFIG["logging"]["training"])



def log_target_distribution(y):
    values, counts = np.unique(y, return_counts=True)
    dist = dict(zip(values, counts))
    logger.info(f"Target distribution: {dist}")


def summarize_feature(name, train_series, test_series):
    logger.info(
        f"{name} | "
        f"Train(mean={train_series.mean():.2f}, std={train_series.std():.2f}) | "
        f"Test(mean={test_series.mean():.2f}, std={test_series.std():.2f})"
    )



def main():

    logger.info("===== Training Pipeline Started =====")

    df, data_path = load_training_data()

    logger.info(f"Training dataset used: {data_path}")
    logger.info(f"Training samples: {len(df)}")

    df = run_data_validation(df)

    # Fix numeric column issue
    df["Total Charges"] = (
        df["Total Charges"]
        .replace(" ", np.nan)
        .astype(float)
        .fillna(0)
    )

    df_sorted = df.sort_values("Tenure Months")

    split_index = int(0.8 * len(df_sorted))
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    log_target_distribution(y_train)

    X_train = run_feature_engineering(train_df)
    X_test = run_feature_engineering(test_df)

    feature_schema = list(X_train.columns)
    logger.info(f"Feature schema captured ({len(feature_schema)} features)")

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
        f"Train tenure range: "
        f"{train_df['Tenure Months'].min()} - {train_df['Tenure Months'].max()}"
    )

    logger.info(
        f"Test tenure range: "
        f"{test_df['Tenure Months'].min()} - {test_df['Tenure Months'].max()}"
    )

    reference_path = Path(CONFIG["paths"]["training_reference"])
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(reference_path, index=False)

    logger.info("Training reference data saved.")



    logger.info("Training candidate models...")

    candidate_models = train_candidate_models(X_train, y_train)

    logger.info("Evaluating candidate models...")

    pipeline, experiment_report, metrics = evaluate_candidates(
        candidate_models,
        X_test,
        y_test
    )

    winner_name = experiment_report["winner"]

    logger.info(f"Champion model selected: {winner_name}")

    model_dir = (
        Path(CONFIG["paths"]["experiments_dir"])
        / f"churn_model_{MODEL_VERSION}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"Model saved at {model_path}")

    # Save experiment comparison report
    with open(model_dir / "experiment_report.json", "w") as f:
        json.dump(experiment_report, f, indent=2)

    logger.info("Experiment report saved.")


    metadata = {
        "model_version": MODEL_VERSION,
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "model_type": winner_name,
        "split_strategy": "time-aware (tenure-based)",
        "feature_schema": feature_schema,
        "feature_count": len(feature_schema),
        "metrics": metrics,
        "dataset": str(data_path),
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved.")
    logger.info("===== Training Pipeline Completed =====")


if __name__ == "__main__":
    main()