"""
Prediction storage module.

Stores inference requests and model outputs so they 
can later be used for monitoring and retraining.
"""

import pandas as pd
from pathlib import Path
from datetime import date, datetime, timezone
from churn_system.schema import REQUIRED_COLUMNS

LOG_PATH = Path("data/inference_logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)

FILE_PATH = LOG_PATH / "predictions.csv"

LOG_COLUMNS = sorted(list(REQUIRED_COLUMNS)) + [
    "prediction_probability",
    "prediction",
    "timestamp",
]

def store_prediction(input_df: pd.DataFrame,
                     probability: float,
                     prediction: int) -> None:
    """
    Store a model inference result for monitoring and retraining.

    Appends the input features, predicted probability, final prediction,
    and a UTC timestamp to a persistent CSV log. The stored records can
    later be used for performance analysis, drift detection, and
    supervised retraining.

    Parameters

    input_df : pd.DataFrame
        Validated model input features.
    probability : float
        Predicted probability for the positive class.
    prediction : int
        Final binary prediction.
    """
    record = input_df.copy()
    
    record["prediction_probability"] = probability
    record["prediction"] = prediction
    record["timestamp"] = datetime.now(timezone.utc)
    
    record = record.reindex(columns=LOG_COLUMNS)
    
    if FILE_PATH.exists():
        record.to_csv(FILE_PATH, mode = "a", header = False, index = False)
    else:
        record.to_csv(FILE_PATH, index = False)
        
        
    
