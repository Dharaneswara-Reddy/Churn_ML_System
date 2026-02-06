import os
import pandas as pd
import numpy as np
import pickle

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



from schema import TARGET_COLUMN, REQUIRED_COLUMNS, ALLOWED_TARGET_VALUES


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
    

def main():
    df = load_data("data/Telco_customer_churn_raw.csv")
    validate_data(df)
    
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
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    
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

    
    os.makedirs("models",exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
        
if __name__ == "__main__":
    main()
    