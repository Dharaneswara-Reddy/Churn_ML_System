"""
Tests for the retraining data loop.

The loop was previously open at both ends: the dataset was written but never read,
and labelled production rows could never be merged into it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn_system.events.predictions import record_label, store_prediction_event
from churn_system.new_data.retraining_data import build_retraining_dataset
from churn_system.training.steps.data_ingestion import resolve_training_data_path

BASE_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges", "Churn Value"]


@pytest.fixture
def base_dataset(isolated_paths):
    frame = pd.DataFrame(
        {
            "Tenure Months": [1, 2, 3],
            "Monthly Charges": [10.0, 20.0, 30.0],
            "Total Charges": [10.0, 40.0, 90.0],
            "Churn Value": [0, 1, 0],
        }
    )
    frame.to_csv(isolated_paths["raw_data"], index=False)
    return frame


@pytest.fixture
def contract(monkeypatch):
    """Model contract stub so event writes do not need a real bundle."""
    import churn_system.events.predictions as ep

    monkeypatch.setattr(ep, "load_model_contract", lambda: {"model_version": "v1"})


class TestIngestionSource:
    def test_prefers_retraining_dataset_when_present(self, isolated_paths, base_dataset):
        assert resolve_training_data_path() == isolated_paths["raw_data"]

        build_retraining_dataset()

        assert resolve_training_data_path() == isolated_paths["retraining_data"]

    def test_falls_back_to_raw_dataset(self, isolated_paths, base_dataset):
        assert resolve_training_data_path() == isolated_paths["raw_data"]


class TestLabelledRowsAreIncorporated:
    def test_unlabelled_predictions_are_not_used(
        self, isolated_paths, base_dataset, contract
    ):
        """A prediction without ground truth has no target and cannot be trained on."""
        store_prediction_event(
            request_id="req-1",
            raw_features={"Tenure Months": 5, "Monthly Charges": 50.0, "Total Charges": 250.0},
            probability=0.8,
            prediction=1,
            latency_seconds=0.01,
        )

        build_retraining_dataset()
        combined = pd.read_csv(isolated_paths["retraining_data"])

        assert len(combined) == len(base_dataset)

    def test_labelled_predictions_are_appended_with_their_target(
        self, isolated_paths, base_dataset, contract
    ):
        """
        The whole point of feedback: a labelled prediction becomes a training row.

        Before this, the merge required an exact column match that redaction made
        impossible, so production rows were always discarded.
        """
        store_prediction_event(
            request_id="req-1",
            raw_features={"Tenure Months": 5, "Monthly Charges": 50.0, "Total Charges": 250.0},
            probability=0.8,
            prediction=1,
            latency_seconds=0.01,
        )
        assert record_label("req-1", 1) is True

        build_retraining_dataset()
        combined = pd.read_csv(isolated_paths["retraining_data"])

        assert len(combined) == len(base_dataset) + 1
        assert list(combined.columns) == BASE_COLUMNS
        assert combined["Churn Value"].iloc[-1] == 1
        assert combined["Tenure Months"].iloc[-1] == 5


class TestLabelRecording:
    def test_unknown_request_id_reports_no_match(self, contract):
        assert record_label("never-served", 1) is False

    def test_invalid_label_rejected(self, contract):
        with pytest.raises(ValueError):
            record_label("req-1", 7)
