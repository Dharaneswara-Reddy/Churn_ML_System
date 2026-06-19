"""Unit tests for data validation (Pandera schema enforcement)."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_system.schema import (
    ALLOWED_TARGET_VALUES,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    validate_training_data,
)


def _make_valid_df(n: int = 5) -> pd.DataFrame:
    """Create a minimal valid training dataframe."""
    import random

    rng = random.Random(42)
    rows = []
    for i in range(n):
        rows.append({
            "CustomerID": f"C{i}",
            "Count": 1,
            "Country": "US",
            "State": "CA",
            "City": "LA",
            "Zip Code": "90001",
            "Lat Long": "34.0, -118.0",
            "Latitude": 34.0,
            "Longitude": -118.0,
            "Gender": rng.choice(["Male", "Female"]),
            "Senior Citizen": "No",
            "Partner": "Yes",
            "Dependents": "No",
            "Tenure Months": rng.randint(1, 72),
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
            "Churn Label": "Yes",
            TARGET_COLUMN: rng.choice([0, 1]),
            "Churn Score": 80,
            "CLTV": 3000,
            "Churn Reason": "Price",
        })
    return pd.DataFrame(rows)


class TestValidateTrainingData:
    """Tests for training data schema enforcement."""

    def test_valid_data_passes(self):
        df = _make_valid_df()
        validate_training_data(df)  # Should not raise

    def test_missing_column_raises(self):
        df = _make_valid_df()
        df = df.drop(columns=["Gender"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_training_data(df)

    def test_invalid_target_values_raises(self):
        df = _make_valid_df()
        df[TARGET_COLUMN] = [0, 1, 2, 0, 1]  # 2 is invalid
        with pytest.raises(ValueError, match="Invalid target values"):
            validate_training_data(df)

    def test_all_required_columns_present_in_constant(self):
        """Sanity check that REQUIRED_COLUMNS contains expected columns."""
        assert "CustomerID" in REQUIRED_COLUMNS
        assert TARGET_COLUMN in REQUIRED_COLUMNS
        assert len(REQUIRED_COLUMNS) == 33

    def test_allowed_target_values(self):
        assert ALLOWED_TARGET_VALUES == {0, 1}
