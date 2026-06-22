"""Tests for data quality and calibration monitoring modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_system.monitoring.calibration import (
    compute_class_balance,
    compute_confidence_distribution,
    compute_expected_calibration_error,
    compute_gini_coefficient,
    compute_prediction_entropy,
)
from churn_system.monitoring.data_quality import (
    check_cardinality,
    check_missing_values,
    check_schema_drift,
    detect_outliers,
)

# ── Data Quality Tests ───────────────────────────────────────────────────────


class TestMissingValues:
    def test_no_missing_values(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = check_missing_values(df)
        assert result["a"] == 0.0
        assert result["b"] == 0.0

    def test_some_missing_values(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6]})
        result = check_missing_values(df)
        assert abs(result["a"] - 1 / 3) < 0.01
        assert abs(result["b"] - 2 / 3) < 0.01


class TestOutlierDetection:
    def test_no_outliers_in_normal_data(self):
        df = pd.DataFrame({"x": np.random.normal(50, 5, 100)})
        result = detect_outliers(df)
        assert "x" in result
        assert result["x"]["outlier_ratio"] < 0.15  # Normal data has ~0.7% outliers

    def test_extreme_outliers_detected(self):
        data = list(np.random.normal(50, 5, 98)) + [500, -500]
        df = pd.DataFrame({"x": data})
        result = detect_outliers(df)
        assert result["x"]["outlier_count"] >= 2


class TestCardinalityCheck:
    def test_detects_new_categories(self):
        ref = pd.DataFrame({"cat": ["A", "B", "C"]})
        prod = pd.DataFrame({"cat": ["A", "B", "D"]})
        result = check_cardinality(prod, ref)
        assert "D" in result["cat"]["new_categories"]
        assert "C" in result["cat"]["missing_categories"]


class TestSchemaDrift:
    def test_detects_new_and_missing_columns(self):
        ref = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        prod = pd.DataFrame({"a": [1], "b": [2], "d": [4]})
        result = check_schema_drift(prod, ref)
        assert "d" in result["new_columns"]
        assert "c" in result["missing_columns"]


# ── Calibration Tests ────────────────────────────────────────────────────────


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_model(self):
        """A model that predicts exactly the true rate should have ECE ≈ 0."""
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        actuals = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = compute_expected_calibration_error(probs, actuals, n_bins=5)
        assert result["ece"] < 0.3  # Should be reasonably low

    def test_badly_calibrated_model(self):
        """A model that always predicts 0.9 but half are negative should have high ECE."""
        probs = np.full(100, 0.9)
        actuals = np.concatenate([np.zeros(50), np.ones(50)])
        result = compute_expected_calibration_error(probs, actuals, n_bins=10)
        assert result["ece"] > 0.3


class TestConfidenceDistribution:
    def test_returns_expected_keys(self):
        probs = np.random.uniform(0, 1, 100)
        result = compute_confidence_distribution(probs)
        assert "mean_confidence" in result
        assert "median_confidence" in result
        assert "low_confidence_ratio" in result
        assert "high_confidence_ratio" in result


class TestPredictionEntropy:
    def test_low_entropy_for_decisive_predictions(self):
        """Predictions near 0 or 1 should have low entropy."""
        probs = np.array([0.01, 0.02, 0.98, 0.99])
        entropy = compute_prediction_entropy(probs)
        assert entropy < 0.2

    def test_high_entropy_for_uncertain_predictions(self):
        """Predictions near 0.5 should have high entropy."""
        probs = np.array([0.48, 0.49, 0.50, 0.51, 0.52])
        entropy = compute_prediction_entropy(probs)
        assert entropy > 0.9


class TestGiniCoefficient:
    def test_uniform_predictions_low_gini(self):
        """All same predictions → Gini near 0."""
        probs = np.full(100, 0.5)
        gini = compute_gini_coefficient(probs)
        assert abs(gini) < 0.1

    def test_spread_predictions_higher_gini(self):
        """Spread predictions → non-zero Gini."""
        probs = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
        gini = compute_gini_coefficient(probs)
        assert abs(gini) > 0.0


class TestClassBalance:
    def test_balanced_predictions(self):
        probs = np.concatenate([np.full(50, 0.3), np.full(50, 0.7)])
        result = compute_class_balance(probs, threshold=0.5)
        assert result["predicted_positive_rate"] == 0.5
        assert result["predicted_negative_rate"] == 0.5

    def test_shift_detection(self):
        probs = np.full(100, 0.8)  # All positive
        result = compute_class_balance(probs, threshold=0.5, reference_positive_rate=0.3)
        assert result["shift_alert"] is True
        assert result["class_balance_shift"] > 0.5
