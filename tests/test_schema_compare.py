"""Unit tests for schema comparison and feature schema validation."""

from __future__ import annotations

import json

import pytest

from churn_system.lifecycle.schema_compare import compare_feature_schemas


@pytest.fixture
def prod_metadata(tmp_path):
    """Create a production metadata file."""
    meta = {
        "feature_schema": ["Country", "State", "City", "Tenure Months", "Monthly Charges"],
        "metrics": {"roc_auc": 0.85},
    }
    path = tmp_path / "prod_metadata.json"
    path.write_text(json.dumps(meta))
    return path


@pytest.fixture
def identical_metadata(tmp_path):
    """Create a challenger metadata with identical schema."""
    meta = {
        "feature_schema": ["Country", "State", "City", "Tenure Months", "Monthly Charges"],
        "metrics": {"roc_auc": 0.87},
    }
    path = tmp_path / "challenger_identical.json"
    path.write_text(json.dumps(meta))
    return path


@pytest.fixture
def added_feature_metadata(tmp_path):
    """Create a challenger metadata with an added feature."""
    meta = {
        "feature_schema": [
            "Country", "State", "City", "Tenure Months", "Monthly Charges", "Total Charges",
        ],
        "metrics": {"roc_auc": 0.88},
    }
    path = tmp_path / "challenger_added.json"
    path.write_text(json.dumps(meta))
    return path


@pytest.fixture
def removed_feature_metadata(tmp_path):
    """Create a challenger metadata with a removed feature (breaking change)."""
    meta = {
        "feature_schema": ["Country", "State", "Tenure Months", "Monthly Charges"],
        "metrics": {"roc_auc": 0.88},
    }
    path = tmp_path / "challenger_removed.json"
    path.write_text(json.dumps(meta))
    return path


class TestSchemaCompare:
    """Feature schema compatibility tests."""

    def test_identical_schemas(self, prod_metadata, identical_metadata):
        result = compare_feature_schemas(prod_metadata, identical_metadata)
        assert result["is_identical"] is True
        assert result["added_features"] == []
        assert result["removed_features"] == []

    def test_added_features_detected(self, prod_metadata, added_feature_metadata):
        result = compare_feature_schemas(prod_metadata, added_feature_metadata)
        assert result["is_identical"] is False
        assert "Total Charges" in result["added_features"]
        assert result["removed_features"] == []

    def test_removed_features_detected(self, prod_metadata, removed_feature_metadata):
        result = compare_feature_schemas(prod_metadata, removed_feature_metadata)
        assert result["is_identical"] is False
        assert "City" in result["removed_features"]

    def test_both_added_and_removed(self, tmp_path):
        prod = {
            "feature_schema": ["A", "B", "C"],
        }
        challenger = {
            "feature_schema": ["B", "C", "D"],
        }
        prod_path = tmp_path / "prod.json"
        chal_path = tmp_path / "chal.json"
        prod_path.write_text(json.dumps(prod))
        chal_path.write_text(json.dumps(challenger))

        result = compare_feature_schemas(prod_path, chal_path)
        assert "A" in result["removed_features"]
        assert "D" in result["added_features"]
        assert result["is_identical"] is False
