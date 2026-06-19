"""
Build Retraining dataset by combining
original training data with production data
"""

from pathlib import Path

import pandas as pd

from churn_system.config.config import CONFIG

RAW_DATA = Path(CONFIG["paths"]["raw_data"])
PROD_LOGS = Path(CONFIG["paths"]["prediction_log_csv"])

OUTPUT = Path(CONFIG["paths"]["retraining_data"])

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
                "timestamp",
                "request_id",
            ],
            errors="ignore",
        )

        if set(prod_df.columns) != set(base_df.columns):
            print(
                "Production log columns do not match training data; "
                "skipping production rows (expected after PII redaction)."
            )
            combined = base_df
        else:
            prod_df = prod_df[base_df.columns]
            combined = pd.concat([base_df, prod_df], ignore_index=True)
            print(f"Added {len(prod_df)} production samples.")
    else:
        combined = base_df
        print("No production data yet.")

    combined.to_csv(OUTPUT, index = False)
    print("Retraining dataset created.")
