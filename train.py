import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from schema import TARGET_COLUMN, REQUIRED_COLUMNS, ALLOWED_TARGET_VALUES


def load_data(path):
    return pd.read_csv(path)

def validate_data(df):
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not set(df[TARGET_COLUMN].unique()).issubset(ALLOWED_TARGET_VALUES):
        raise ValueError(f"Invalid target values found : {df[TARGET_COLUMN].unique()}")

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
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    
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
            ("model", LogisticRegression(max_iter=1000))
        ]
    )
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    
    pipeline.fit(X_train,y_train)
    
    os.makedirs("models",exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
        
if __name__ == "__main__":
    main()
    