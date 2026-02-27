from pathlib import Path
import json

TARGET_COLUMN = "Churn Value"

ALLOWED_TARGET_VALUES = {0, 1}



REQUIRED_COLUMNS = {
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
}


def validate_training_data(df):
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not set(df[TARGET_COLUMN].unique()).issubset(ALLOWED_TARGET_VALUES):
        raise ValueError(
            f"Invalid target values found: {df[TARGET_COLUMN].unique()}"
        )



PRODUCTION_METADATA = Path(
    "models/production/current/current/metadata.json"
)


def load_feature_schema():
    """
    Load feature schema from deployed production model metadata.
    """
    if not PRODUCTION_METADATA.exists():
        raise FileNotFoundError(
            "Production metadata not found. Cannot validate inference schema."
        )

    with open(PRODUCTION_METADATA, "r") as f:
        metadata = json.load(f)

    return set(metadata["feature_schema"])


def validate_inference_data(df):
    """
    Validate inference dataframe against MODEL FEATURE SCHEMA
    (not raw dataset schema).
    """

    required_features = load_feature_schema()

    missing = required_features - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required model features at inference: {missing}"
        )

    if TARGET_COLUMN in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must not appear at inference"
        )

    return df[list(required_features)]