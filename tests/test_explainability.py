"""Tests for the Explainable AI module (SHAP explanations)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_system.explainability.shap_explainer import (
    explain_prediction,
    reset_explainer,
)
from churn_system.serving.model_registry import ModelRegistry


class _StubPipeline:
    """Minimal pipeline stub that mimics sklearn Pipeline interface."""

    class _model:
        pass

    named_steps = {"model": _model}

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.column_stack([
            np.full(n, 0.3),
            np.full(n, 0.7),
        ])


class TestExplainPrediction:
    """Verify that SHAP explanations return the correct structure."""

    def setup_method(self):
        reset_explainer()
        ModelRegistry.reset()

    def teardown_method(self):
        reset_explainer()
        ModelRegistry.reset()

    def test_explain_returns_expected_keys(self, monkeypatch, tmp_path):
        """Explanation result should contain all required keys."""
        # Create a minimal training reference CSV
        ref_data = pd.DataFrame({
            "Tenure Months": np.random.randint(1, 72, 50),
            "Monthly Charges": np.random.uniform(20, 100, 50),
            "Total Charges": np.random.uniform(100, 5000, 50),
        })
        ref_path = tmp_path / "training_reference.csv"
        ref_data.to_csv(ref_path, index=False)

        # Monkeypatch CONFIG to point to our temp training reference
        from churn_system.config.config import CONFIG

        monkeypatch.setitem(
            CONFIG["paths"],
            "training_reference",
            str(ref_path),
        )

        # Mock model registry to return our stub
        stub = _StubPipeline()

        class StubRegistry:
            def get_model(self):
                return stub

        monkeypatch.setattr(
            "churn_system.serving.model_registry.ModelRegistry.instance",
            staticmethod(lambda: StubRegistry()),
        )

        # Mock build_features and validate_inference_data to pass through
        monkeypatch.setattr(
            "churn_system.explainability.shap_explainer.build_features",
            lambda df, training=False: df[["Tenure Months", "Monthly Charges", "Total Charges"]],
        )
        monkeypatch.setattr(
            "churn_system.explainability.shap_explainer.validate_inference_data",
            lambda df: df,
        )

        result = explain_prediction({
            "Tenure Months": 24,
            "Monthly Charges": 65.0,
            "Total Charges": 1500.0,
        })

        assert "prediction_probability" in result
        assert "base_value" in result
        assert "shap_values" in result
        assert "top_positive_drivers" in result
        assert "top_negative_drivers" in result
        assert isinstance(result["shap_values"], dict)
        assert result["prediction_probability"] == 0.7
