import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "CustomerID": "CUST_001",
    "Count": 1,
    "Country": "Unknown",
    "State": "Unknown",
    "City": "Unknown",
    "Zip Code": "000000",
    "Lat Long": "0.0,0.0",
    "Latitude": 0.0,
    "Longitude": 0.0,
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
    "Contract": "Month-to-Month",
    "Paperless Billing": "Yes",
    "Payment Method": "Credit Card",
    "Monthly Charges": 70.5,
    "Total Charges": 850.0,
    "Churn Label": "No",
    "Churn Score": 20,
    "CLTV": 4000,
    "Churn Reason": "Unknown"
}

response = requests.post(url, json=payload)
print(response.json())
