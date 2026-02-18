"""
Model Health Evaluation

Uses drift results to decide whether
model retraining should be triggered.
"""

import json
from pathlib import Path
from churn_system.monitoring.drift import calculate_psi
import pandas as pd
import numpy as np


TRAIN_PATH = Path("data/training_reference.csv")
PROD_PATH = Path("data/inference_logs/predictions.csv")

REPORT_PATH = Path("models/monitoring")
REPORT_PATH.mkdir(parents=True, exist_ok=True)

HEALTH_FILE = REPORT_PATH / "health_report.json"

PSI_THRESHOLD = 0.2
DRIFT_FEATURE_LIMIT = 2


def evaluate_model_health():
    """
    Evaluate model stability using PSI drift metrics.
    """

    if not TRAIN_PATH.exists() or not PROD_PATH.exists():
        print("Missing data for health evaluation.")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    prod_df = pd.read_csv(PROD_PATH)

    numeric_cols = train_df.select_dtypes(include=np.number).columns

    drifting_features = []

    for col in numeric_cols:

        if col not in prod_df.columns:
            continue   

        psi = calculate_psi(
            train_df[col].dropna(),
            prod_df[col].dropna()
        )

        if psi > PSI_THRESHOLD:
            drifting_features.append({
                "feature": col,
                "psi": round(float(psi), 4)
            })

    retrain_required = len(drifting_features) >= DRIFT_FEATURE_LIMIT

    report = {
        "drifting_feature_count": len(drifting_features),
        "drifting_features": drifting_features,
        "retraining_recommended": retrain_required
    }

    with open(HEALTH_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print("\n--- Model Health ---")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    evaluate_model_health()
