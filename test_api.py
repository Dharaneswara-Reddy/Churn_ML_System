"""
Manual smoke script against a running API (uvicorn churn_system.api.api:app).

Uses only model feature columns (see models/production/current/metadata.json).
Optional: export CHURN_API_KEY and pass header X-API-Key.
"""

import os

import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "Country": "US",
    "State": "CA",
    "City": "TestCity",
    "Zip Code": "90210",
    "Lat Long": "34.0, -118.0",
    "Latitude": 34.0,
    "Longitude": -118.0,
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber Optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 70.5,
    "Total Charges": 850.0,
}

headers = {}
key = os.environ.get("CHURN_API_KEY")
if key:
    headers["X-API-Key"] = key

response = requests.post(url, json=payload, headers=headers, timeout=30)
print(response.status_code, response.json())
