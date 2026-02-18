"""
Build Retraining dataset by combining
original training data with production data
"""

import pandas as pd
from pathlib import Path

RAW_DATA = Path("data/Telco_customer_churn_raw.csv")
PROD_LOGS = Path("data/inference_logs/predictions.csv")

OUTPUT = Path("data/retraining_dataset.csv")

def build_retraining_dataset():
    if not RAW_DATA.exists():
        raise ValueError("Original dataset missing.")
    
    base_df = pd.read_csv(RAW_DATA)
    
    if PROD_LOGS.exists():
        prod_df = pd.read_csv(PROD_LOGS)
        
        prod_df = prod_df.drop(
            columns=[
                "prediction",
                "prediction_probability",
                "timestamp"
            ],
            errors = "ignore"
        )
        combined = pd.concat([base_df, prod_df], ignore_index = True)
        print(f"Added {len(prod_df)} production samples.")
    else:
        combined = base_df
        print("No production data yet.")
        
    combined.to_csv(OUTPUT, index = False)
    print("Retraining dataset created.")