# schema.py

TARGET_COLUMN = "Churn Value"

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
    "Churn Reason"
}

ALLOWED_TARGET_VALUES = {0, 1}


def validate_training_data(df):
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not set(df[TARGET_COLUMN].unique()).issubset(ALLOWED_TARGET_VALUES):
        raise ValueError(
            f"Invalid target values found: {df[TARGET_COLUMN].unique()}"
        )


def validate_inference_data(df):
    """
    Enforce the SAME feature contract used during training.
    If the model saw a feature during training, it must exist at inference.
    """

    required_features = REQUIRED_COLUMNS - {TARGET_COLUMN}

    missing = required_features - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features at inference: {missing}")

    if TARGET_COLUMN in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must not be present at inference time"
        )

    # Keep exact column order irrelevant (sklearn handles by name)
    return df[list(required_features)]
