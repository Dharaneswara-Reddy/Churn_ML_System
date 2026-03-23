#!/usr/bin/env python3
"""
Generate a small synthetic CSV for CI smoke training (no real customer data).

Usage:
  python scripts/generate_smoke_csv.py /path/to/out.csv
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from churn_system.schema import REQUIRED_COLUMNS, TARGET_COLUMN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, help="Output CSV path")
    parser.add_argument("--rows", type=int, default=400)
    args = parser.parse_args()

    rng = random.Random(42)
    np.random.seed(42)

    n = args.rows
    rows = []

    cats = {
        "Gender": ["Male", "Female"],
        "Senior Citizen": ["Yes", "No"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "Phone Service": ["Yes", "No"],
        "Multiple Lines": ["Yes", "No"],
        "Internet Service": ["DSL", "Fiber Optic", "No"],
        "Online Security": ["Yes", "No", "No internet service"],
        "Online Backup": ["Yes", "No", "No internet service"],
        "Device Protection": ["Yes", "No", "No internet service"],
        "Tech Support": ["Yes", "No", "No internet service"],
        "Streaming TV": ["Yes", "No", "No internet service"],
        "Streaming Movies": ["Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Paperless Billing": ["Yes", "No"],
        "Payment Method": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "Churn Label": ["Yes", "No"],
    }

    for i in range(n):
        tenure = rng.randint(0, 72)
        monthly = round(rng.uniform(20, 120), 2)
        total = round(monthly * tenure + rng.uniform(0, 200), 2)
        churn_val = int(rng.random() < 0.27)

        row = {
            "CustomerID": f"CUST_{i:05d}",
            "Count": 1,
            "Country": "US",
            "State": "CA",
            "City": "TestCity",
            "Zip Code": f"{rng.randint(90000, 99999):05d}",
            "Lat Long": "0.0, 0.0",
            "Latitude": round(rng.uniform(25, 49), 6),
            "Longitude": round(rng.uniform(-125, -66), 6),
            "Gender": rng.choice(cats["Gender"]),
            "Senior Citizen": rng.choice(cats["Senior Citizen"]),
            "Partner": rng.choice(cats["Partner"]),
            "Dependents": rng.choice(cats["Dependents"]),
            "Tenure Months": tenure,
            "Phone Service": rng.choice(cats["Phone Service"]),
            "Multiple Lines": rng.choice(cats["Multiple Lines"]),
            "Internet Service": rng.choice(cats["Internet Service"]),
            "Online Security": rng.choice(cats["Online Security"]),
            "Online Backup": rng.choice(cats["Online Backup"]),
            "Device Protection": rng.choice(cats["Device Protection"]),
            "Tech Support": rng.choice(cats["Tech Support"]),
            "Streaming TV": rng.choice(cats["Streaming TV"]),
            "Streaming Movies": rng.choice(cats["Streaming Movies"]),
            "Contract": rng.choice(cats["Contract"]),
            "Paperless Billing": rng.choice(cats["Paperless Billing"]),
            "Payment Method": rng.choice(cats["Payment Method"]),
            "Monthly Charges": monthly,
            "Total Charges": total,
            "Churn Label": rng.choice(cats["Churn Label"]),
            TARGET_COLUMN: churn_val,
            "Churn Score": rng.randint(0, 100),
            "CLTV": rng.randint(1000, 8000),
            "Churn Reason": "Unknown",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
