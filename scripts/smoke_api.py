"""
Manual smoke check against a running API (uvicorn churn_system.api.api:app).

Uses only model feature columns (see models/production/current/metadata.json).
Optional: export CHURN_API_KEY and the request will send it as X-API-Key.

    python scripts/smoke_api.py [--url http://127.0.0.1:8000/predict]

This is a script, not a pytest module — it performs real network I/O and is
deliberately kept out of tests/ so collection never tries to import it.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

DEFAULT_URL = "http://127.0.0.1:8000/predict"

PAYLOAD = {
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"predict endpoint (default: {DEFAULT_URL})")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    headers = {}
    key = os.environ.get("CHURN_API_KEY")
    if key:
        headers["X-API-Key"] = key

    try:
        response = requests.post(args.url, json=PAYLOAD, headers=headers, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    print(response.status_code, response.text)
    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
