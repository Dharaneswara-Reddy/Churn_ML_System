# schema.py

# Column names follow the source dataset exactly (Telco Churn CSV)

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


