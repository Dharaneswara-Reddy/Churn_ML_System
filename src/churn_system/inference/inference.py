import pickle
import pandas as pd
import json
from pathlib import Path

from churn_system.schema import validate_inference_data

# Load trained pipeline
with open(
    "models/experiments/churn_model_v1/model.pkl", "rb"
) as f:
    model = pickle.load(f)

# FULL inference payload (must match training features)
data = {
    "CustomerID": ["CUST_001"],
    "Count": [1],
    "Country": ["Unknown"],
    "State": ["Unknown"],
    "City": ["Unknown"],
    "Zip Code": ["000000"],
    "Lat Long": ["0.0,0.0"],
    "Latitude": [0.0],
    "Longitude": [0.0],
    "Gender": ["Male"],
    "Senior Citizen": ["No"],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "Tenure Months": [12],
    "Phone Service": ["Yes"],
    "Multiple Lines": ["No"],
    "Internet Service": ["Fiber Optic"],
    "Online Security": ["No"],
    "Online Backup": ["Yes"],
    "Device Protection": ["No"],
    "Tech Support": ["No"],
    "Streaming TV": ["Yes"],
    "Streaming Movies": ["Yes"],
    "Contract": ["Month-to-Month"],
    "Paperless Billing": ["Yes"],
    "Payment Method": ["Credit Card"],
    "Monthly Charges": [70.5],
    "Total Charges": [850.0],
    "Churn Label": ["No"],
    "Churn Score": [20],
    "CLTV": [4000],
    "Churn Reason": ["Unknown"]
}


def load_feature_contract():
    metadata_path = Path("models/production/current/metadata.json")

    if not metadata_path.exists():
        raise RuntimeError("Metadata file missing in production model")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata["feature_schema"]

df = pd.DataFrame(data)


df_valid = validate_inference_data(df)

expected_features = load_feature_contract()

incoming_features = list(df_valid.columns)

if incoming_features != expected_features:
    missing = set(expected_features) - set(incoming_features)
    extra = set(incoming_features) - set(expected_features)

    raise ValueError(
        f"Feature schema mismatch | Missing: {missing} | Extra: {extra}"
    )


prob = model.predict_proba(df_valid)[:, 1]
print(f"Churn probability: {prob[0]:.4f}")

