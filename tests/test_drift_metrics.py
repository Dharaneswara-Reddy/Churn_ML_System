"""
Drift observability tests.

The defect: ``prometheus_client``'s registry is per-process. Drift gauges were set
in the *scheduler* process while Prometheus scrapes only the API, so
``churn_drifting_feature_count`` could never appear on ``/metrics`` — and the
``ChurnModelDriftDetected`` alert rule keyed on it was structurally incapable of
firing. Not flaky: impossible, and silently so.

These tests assert the state actually reaches the scrape output, which is the only
thing that makes the alert real.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(isolated_paths, monkeypatch):
    import churn_system.api.api as api_mod

    return TestClient(api_mod.app), isolated_paths


def _write_health_report(paths, *, drifting=3, retrain=True):
    report = {
        "status": "evaluated",
        "drifting_feature_count": drifting,
        "retraining_recommended": retrain,
        "drifting_features": [],
    }
    (paths["monitoring_dir"] / "health_report.json").write_text(json.dumps(report))


class TestDriftReachesTheScrapeEndpoint:
    def test_drift_count_appears_in_metrics_output(self, api_client):
        """
        The core regression: a value produced by the scheduler must be visible to
        the process Prometheus actually scrapes.
        """
        client, paths = api_client
        _write_health_report(paths, drifting=3, retrain=True)

        body = client.get("/metrics").text

        assert "churn_drifting_feature_count" in body
        assert "churn_drifting_feature_count 3.0" in body

    def test_retraining_recommendation_appears(self, api_client):
        client, paths = api_client
        _write_health_report(paths, drifting=2, retrain=True)

        body = client.get("/metrics").text

        assert "churn_retraining_recommended 1.0" in body

    def test_alert_threshold_is_actually_observable(self, api_client):
        """
        alert_rules.yml fires on `churn_drifting_feature_count >= 2`. Prove the
        series both exists and crosses that threshold when drift is present.
        """
        client, paths = api_client
        _write_health_report(paths, drifting=5, retrain=True)

        body = client.get("/metrics").text
        line = next(
            ln
            for ln in body.splitlines()
            if ln.startswith("churn_drifting_feature_count ")
        )
        value = float(line.split()[-1])

        assert value >= 2

    def test_metrics_survive_a_missing_health_report(self, api_client):
        """
        A missing state file must degrade one gauge, never break the whole scrape —
        an unscrapeable endpoint takes every other metric down with it.
        """
        client, _ = api_client

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "churn_api_requests_total" in response.text

    def test_metrics_survive_a_corrupt_health_report(self, api_client):
        client, paths = api_client
        (paths["monitoring_dir"] / "health_report.json").write_text("{not json")

        assert client.get("/metrics").status_code == 200


class TestPreviouslyDeadGauges:
    """
    GINI_COEFFICIENT, PREDICTED_POSITIVE_RATE and PREDICTED_NEGATIVE_RATE were
    declared but never set anywhere — permanent zeros on /metrics, which is worse
    than absent because a dashboard built on them looks healthy.
    """

    def test_gini_is_derived_from_the_champion(self, api_client):
        client, paths = api_client
        bundle = paths["production_model"].parent
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "metadata.json").write_text(
            json.dumps({"model_version": "v9", "metrics": {"roc_auc": 0.85}})
        )

        body = client.get("/metrics").text

        # Gini is exactly 2*AUC - 1, not an approximation.
        assert "churn_model_gini_coefficient 0.7" in body

    def test_champion_version_is_exposed_as_a_label(self, api_client):
        client, paths = api_client
        bundle = paths["production_model"].parent
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "metadata.json").write_text(
            json.dumps({"model_version": "v9", "metrics": {}})
        )

        body = client.get("/metrics").text

        assert 'churn_champion_model_info{model_version="v9"}' in body


class TestOutboxBacklogIsExposed:
    def test_backlog_by_status_appears(self, api_client):
        from churn_system.events.db import (
            OutboxEvent,
            OutboxStatus,
            SessionLocal,
            init_db,
            now_utc,
        )

        client, _ = api_client
        init_db()
        with SessionLocal() as session:
            session.add(
                OutboxEvent(
                    created_at=now_utc(),
                    event_type="prediction_made",
                    payload={},
                    status=OutboxStatus.DEAD_LETTER.value,
                    attempts=5,
                )
            )
            session.commit()

        body = client.get("/metrics").text

        assert 'churn_outbox_backlog{status="DEAD_LETTER"} 1.0' in body
