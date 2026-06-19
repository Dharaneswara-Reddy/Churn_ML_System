"""Unit tests for feature engineering and feature type inference."""

from __future__ import annotations

import pandas as pd

from churn_system.features.build_features import TARGET_COLUMN, build_features
from churn_system.training.feature_types import infer_feature_types


class TestBuildFeatures:
    """Tests for the shared feature builder."""

    def _make_raw_df(self, include_target: bool = True):
        """Create a minimal raw dataframe matching expected schema."""
        data = {
            "CustomerID": ["CUST_001"],
            "Count": [1],
            "Country": ["US"],
            "State": ["CA"],
            "City": ["LA"],
            "Zip Code": ["90001"],
            "Lat Long": ["34.0, -118.0"],
            "Latitude": [34.0],
            "Longitude": [-118.0],
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
            "Contract": ["Month-to-month"],
            "Paperless Billing": ["Yes"],
            "Payment Method": ["Electronic check"],
            "Monthly Charges": [70.5],
            "Total Charges": ["850.0"],
            "Churn Label": ["Yes"],
            "Churn Score": [80],
            "CLTV": [3000],
            "Churn Reason": ["Price"],
        }
        if include_target:
            data[TARGET_COLUMN] = [1]
        return pd.DataFrame(data)

    def test_drops_target_column(self):
        df = self._make_raw_df(include_target=True)
        result = build_features(df, training=True)
        assert TARGET_COLUMN not in result.columns

    def test_drops_meta_columns(self):
        df = self._make_raw_df()
        result = build_features(df)
        for col in ["CustomerID", "Count", "Churn Label", "Churn Score", "Churn Reason", "CLTV"]:
            assert col not in result.columns

    def test_total_charges_coerced_to_numeric(self):
        df = self._make_raw_df()
        df["Total Charges"] = " "  # raw data quirk
        result = build_features(df)
        assert result["Total Charges"].dtype == float
        assert result["Total Charges"].iloc[0] == 0.0

    def test_does_not_modify_original(self):
        df = self._make_raw_df()
        original_cols = set(df.columns)
        build_features(df)
        assert set(df.columns) == original_cols  # original untouched

    def test_inference_mode_works_without_target(self):
        df = self._make_raw_df(include_target=False)
        result = build_features(df, training=False)
        assert TARGET_COLUMN not in result.columns


class TestInferFeatureTypes:
    """Tests for feature type inference."""

    def test_int_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = infer_feature_types(df)
        assert result["x"] == "int"

    def test_float_column(self):
        df = pd.DataFrame({"x": [1.0, 2.5, 3.7]})
        result = infer_feature_types(df)
        assert result["x"] == "float"

    def test_str_column(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        result = infer_feature_types(df)
        assert result["x"] == "str"

    def test_bool_column(self):
        df = pd.DataFrame({"x": [True, False, True]})
        result = infer_feature_types(df)
        assert result["x"] == "bool"

    def test_mixed_columns(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "score": [0.95, 0.87],
            "active": [True, False],
        })
        result = infer_feature_types(df)
        assert result == {"name": "str", "age": "int", "score": "float", "active": "bool"}
