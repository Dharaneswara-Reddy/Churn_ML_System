# schema.py

# Column names follow the source dataset exactly (Telco Churn CSV)

from numpy import require


REQUIRED_COLUMNS = {
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
    "Churn Value",
}

TARGET_COLUMN = "Churn Value"
ALLOWED_TARGET_VALUES = {0, 1}

def validate_inference_data(df):
    """
    Validate input data at inference time.
    Ensures required features exist and target column in absent.
    """
    
    required_features = REQUIRED_COLUMNS - {TARGET_COLUMN}
    
    missing = required_features - set(df.columns)
    if missing:
        raise ValueError(f"Missing requierd features at inference : {missing}")
    
    if TARGET_COLUMN in df.columns:
        raise ValueError(f"Target Column '{TARGET_COLUMN}' should not be present at inference time")
    
    extra = set(df.columns) - required_features
    if extra:
        print(f"Warning : extra columns ignored at inference : {extra}")
        
    return df[list(required_features)]
