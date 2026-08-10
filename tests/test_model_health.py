"""
Regression tests for the retraining decision.

``evaluate_model_health`` drives the automated retrain -> promote cycle, so a false
positive here is not a bad report — it is an unnecessary model swap in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_system.monitoring.model_health import evaluate_model_health


@pytest.fixture
def reference_and_production(isolated_paths, monkeypatch):
    """Write a training reference and return a writer for the production feed."""
    # Drive the decision purely from the CSV feed this fixture controls. Otherwise
    # any prediction event left in the shared test database would be picked up as
    # "production data" and make these assertions order-dependent.
    monkeypatch.setattr(
        "churn_system.monitoring.prediction_reader.load_predictions_df",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "Tenure Months": rng.integers(1, 72, 2000),
            "Monthly Charges": rng.normal(70, 20, 2000),
            "Total Charges": rng.normal(2000, 800, 2000),
        }
    )
    reference.to_csv(isolated_paths["training_reference"], index=False)

    def write_production(df: pd.DataFrame) -> None:
        path = isolated_paths["prediction_log_csv"]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    return reference, write_production


class TestInsufficientData:
    """Small samples must not be mistaken for drift."""

    def test_handful_of_predictions_does_not_trigger_retraining(
        self, reference_and_production
    ):
        """
        Five rows drawn from the reference itself have zero drift by construction.

        Before the minimum-sample guard, PSI against such a small sample exceeded the
        threshold on every feature and recommended a full retrain.
        """
        reference, write_production = reference_and_production
        write_production(reference.sample(5, random_state=1))

        report = evaluate_model_health()

        assert report["retraining_recommended"] is False
        assert report["status"] == "insufficient_data"
        assert report["drifting_feature_count"] == 0

    def test_missing_production_data_reports_unknown_not_healthy(self, isolated_paths):
        report = evaluate_model_health()

        assert report["status"] == "insufficient_data"
        assert report["retraining_recommended"] is False


class TestDriftDecision:
    """With enough data, real drift is detected and stable data is left alone."""

    def test_stable_production_does_not_recommend_retraining(
        self, reference_and_production
    ):
        reference, write_production = reference_and_production
        write_production(reference.sample(500, random_state=2))

        report = evaluate_model_health()

        assert report["status"] == "evaluated"
        assert report["retraining_recommended"] is False

    def test_shifted_production_recommends_retraining(self, reference_and_production):
        reference, write_production = reference_and_production
        shifted = reference.sample(500, random_state=3).copy()
        shifted["Monthly Charges"] += 200
        shifted["Total Charges"] += 5000
        write_production(shifted)

        report = evaluate_model_health()

        assert report["retraining_recommended"] is True
        assert report["drifting_feature_count"] >= 2

    def test_report_records_which_features_were_evaluated(
        self, reference_and_production
    ):
        """A report that cannot say what it looked at cannot be audited."""
        reference, write_production = reference_and_production
        write_production(reference.sample(500, random_state=4))

        report = evaluate_model_health()

        assert set(report["evaluated_features"]) == set(reference.columns)
