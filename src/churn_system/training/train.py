import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json
from pathlib import Path


from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from churn_system.logging.logger import get_logger
from churn_system.schema import TARGET_COLUMN
from churn_system.config.config import CONFIG
from churn_system.training.steps.data_ingestion import load_training_data
from churn_system.training.steps.data_validation import run_data_validation
from churn_system.training.steps.feature_engineering import run_feature_engineering
from churn_system.training.steps.model_training import train_model
from churn_system.training.steps.model_evaluation import evaluate_model

MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = get_logger(__name__, CONFIG["logging"]["training"])




    
def log_target_distribution(y):
    values, counts = np.unique(y, return_counts = True)
    dist = dict(zip(values,counts))
    print("Target distribtuion:", dist)
    
def summarize_feature(name, train_series, test_series):
    print(f"\nFeature: {name}")
    print(f"Train -> Mean : {train_series.mean():.2f}, std : {train_series.std():.2f}")
    print(f"Test -> Mean : {test_series.mean():.2f}, std : {test_series.std():.2f}")
    
    

def main():
    data_path = Path("data/retraining_dataset.csv")
    if not data_path.exists():
        data_path = "data/Telco_customer_churn_raw.csv"
    df = load_training_data()
    logger.info(f"Training dataset used: {data_path}")
    logger.info(f"Training samples: {len(df)}")
    df = run_data_validation(df)
    
    df["Total Charges"] = pd.to_numeric(df["Total Charges"],errors='coerce')
    df["Total Charges"] = df["Total Charges"].fillna(0)
    
    TARGET = "Churn Value"
    DROP_COLS = [
        "CustomerID",
        "Count",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
        "CLTV"
    ]
    
    y = df["Churn Value"]
    log_target_distribution(y)
    X = run_feature_engineering(df) 
    feature_schema = list(X.columns)
    logger.info(f"Feature schema captured ({len(feature_schema)} features)")
    neg, pos = np.bincount(y)
    ratio = neg / pos
    print(f"Negative/Positive ratio: {ratio:.2f}")

    


    
    df_sorted = df.sort_values("Tenure Months")
    
    split_index = int(0.8 * len(df_sorted))
    
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    
    summarize_feature(
        "Tenure Months",
        X_train["Tenure Months"],
        X_test["Tenure Months"]
    )

    summarize_feature(
        "Monthly Charges",
        X_train["Monthly Charges"],
        X_test["Monthly Charges"]
    )

    summarize_feature(
        "Total Charges",
        X_train["Total Charges"],
        X_test["Total Charges"]
    )

    
    print(f"Train tenure range: {X_train['Tenure Months'].min()} - {X_train['Tenure Months'].max()}")
    print(f"Test tenure range: {X_test['Tenure Months'].min()} - {X_test['Tenure Months'].max()}")

    reference_path = "data/training_reference.csv"
    X_train.to_csv(reference_path, index = False)
    logger.info("Training reference data saved.")
        
    pipeline = train_model(X_train, y_train)
    metrics, probs = evaluate_model(pipeline, X_test, y_test)
    model_dir = f"models/experiments/churn_model_{MODEL_VERSION}"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
        logger.info(f"Model saved at {model_dir}")
        
        
    metadata = {
    "model_version": MODEL_VERSION,
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "split_strategy": "time-aware (tenure-based)",
    "class_weight": "balanced",
    "feature_schema" : feature_schema,
    "feature_count" : len(feature_schema),
    "metrics" : metrics,
    "dataset" : str(data_path)
    }
    
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent = 2)
        
if __name__ == "__main__":
    main()
    