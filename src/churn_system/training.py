import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)



from churn_system.schema import TARGET_COLUMN, REQUIRED_COLUMNS, ALLOWED_TARGET_VALUES

MODEL_VERSION = "v1"

def load_data(path):
    return pd.read_csv(path)

def validate_data(df):
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not set(df[TARGET_COLUMN].unique()).issubset(ALLOWED_TARGET_VALUES):
        raise ValueError(f"Invalid target values found : {df[TARGET_COLUMN].unique()}")
    
def log_target_distribution(y):
    values, counts = np.unique(y, return_counts = True)
    dist = dict(zip(values,counts))
    print("Target distribtuion:", dist)
    
def summarize_feature(name, train_series, test_series):
    print(f"\nFeature: {name}")
    print(f"Train -> Mean : {train_series.mean():.2f}, std : {train_series.std():.2f}")
    print(f"Test -> Mean : {test_series.mean():.2f}, std : {test_series.std():.2f}")
    
    

def main():
    df = load_data("data/Telco_customer_churn_raw.csv")
    validate_data(df)
    
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
    
    y = df[TARGET]
    log_target_distribution(y)
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    
    neg, pos = np.bincount(y)
    ratio = neg / pos
    print(f"Negative/Positive ratio: {ratio:.2f}")

    
    categorical_cols = X.select_dtypes(include = ["object"]).columns
    numerical_cols = X.select_dtypes(exclude = ["object"]).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    
    pipeline = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000,class_weight="balanced"))
        ]
    )
    
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
    print("Training reference data saved.")
        
    pipeline.fit(X_train,y_train)
    
    probs = pipeline.predict_proba(X_test)[:,1]
    preds = pipeline.predict(X_test)
    
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test,preds):.4f}")
    print(f"Recall : {recall_score(y_test,preds):.4f}")
    print(f"F1-Score: {f1_score(y_test,preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test,probs):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test,probs):.4f}")
    
    for t in [0.3, 0.5, 0.7]:
        preds_t = (probs >= t).astype(int)
        print(
            f"Threshold {t} | "
            f"Precision: {precision_score(y_test, preds_t):.3f} | "
            f"Recall: {recall_score(y_test, preds_t):.3f}"
        )

    
    model_dir = f"models/experiments/churn_model_{MODEL_VERSION}"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
        
        
    metadata = {
    "model_version": MODEL_VERSION,
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "split_strategy": "time-aware (tenure-based)",
    "class_weight": "balanced",
    "features_used": list(X_train.columns),
    "notes": "Day 7 model with drift-aware split and cost-sensitive learning"
    }
    
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent = 2)
        
if __name__ == "__main__":
    main()
    