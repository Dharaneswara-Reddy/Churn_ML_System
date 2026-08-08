"""Regression: /predict happy path with mocked model.

Tests monkeypatch ``_get_model`` which now delegates to the
thread-safe ModelRegistry singleton. The stub model replaces the
registry lookup so no real model.pkl is needed on disk.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fake model metadata so the API module can be imported without a real
# model bundle on disk (CI environments don't have models/production/).
# ---------------------------------------------------------------------------
_FAKE_METADATA = {
    "feature_schema": [
        "Country", "State", "City", "Zip Code", "Lat Long",
        "Latitude", "Longitude", "Gender", "Senior Citizen",
        "Partner", "Dependents", "Tenure Months", "Phone Service",
        "Multiple Lines", "Internet Service", "Online Security",
        "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies", "Contract",
        "Paperless Billing", "Payment Method", "Monthly Charges",
        "Total Charges",
    ],
    "feature_count": 26,
    "metrics": {},
    "feature_types": {
        "Country": "str", "State": "str", "City": "str",
        "Zip Code": "str", "Lat Long": "str",
        "Latitude": "float", "Longitude": "float",
        "Gender": "str", "Senior Citizen": "str",
        "Partner": "str", "Dependents": "str",
        "Tenure Months": "int", "Phone Service": "str",
        "Multiple Lines": "str", "Internet Service": "str",
        "Online Security": "str", "Online Backup": "str",
        "Device Protection": "str", "Tech Support": "str",
        "Streaming TV": "str", "Streaming Movies": "str",
        "Contract": "str", "Paperless Billing": "str",
        "Payment Method": "str", "Monthly Charges": "float",
        "Total Charges": "float",
    },
}


def _patch_model_contract(monkeypatch):
    """Patch model contract loader so api.py can be imported without real artifacts."""
    import churn_system.inference.model_contract as mc

    mc.load_model_contract.cache_clear()

    import churn_system.api.schema_generator as sg

    monkeypatch.setattr(sg, "load_model_contract", lambda: _FAKE_METADATA)


@pytest.fixture
def api_module(monkeypatch):
    monkeypatch.setenv("CHURN_DISABLE_RATE_LIMIT", "1")
    monkeypatch.delenv("CHURN_API_KEY", raising=False)
    _patch_model_contract(monkeypatch)
    import churn_system.api.api as api_mod

    importlib.reload(api_mod)
    return api_mod


def _sample_payload():
    return {
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


def _stub_model():
    """Return a lightweight stub that mimics sklearn predict_proba()."""

    class StubModel:
        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))

    return StubModel()


def test_predict_returns_probability(api_module, monkeypatch):
    monkeypatch.setattr(api_module, "_get_model", lambda: _stub_model())

    client = TestClient(api_module.app)
    r = client.post("/predict", json=_sample_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert "churn_probability" in body
    assert "request_id" in body
    assert body["prediction"] in (0, 1)


def test_predict_rejects_extra_field(api_module, monkeypatch):
    monkeypatch.setattr(api_module, "_get_model", lambda: _stub_model())

    p = _sample_payload()
    p["unexpected"] = 1
    client = TestClient(api_module.app)
    r = client.post("/predict", json=p)
    assert r.status_code == 422


def test_predict_requires_api_key_when_set(monkeypatch):
    monkeypatch.setenv("CHURN_DISABLE_RATE_LIMIT", "1")
    monkeypatch.setenv("CHURN_API_KEY", "secret")
    _patch_model_contract(monkeypatch)
    import churn_system.api.api as api_mod

    importlib.reload(api_mod)

    monkeypatch.setattr(api_mod, "_get_model", lambda: _stub_model())

    client = TestClient(api_mod.app)
    r = client.post("/predict", json=_sample_payload())
    assert r.status_code == 401

    r2 = client.post(
        "/predict",
        json=_sample_payload(),
        headers={"X-API-Key": "secret"},
    )
    assert r2.status_code == 200


def test_batch_predict_returns_multiple_results(api_module, monkeypatch):
    monkeypatch.setattr(api_module, "_get_model", lambda: _stub_model())

    client = TestClient(api_module.app)
    batch = [_sample_payload(), _sample_payload()]
    r = client.post("/predict/batch", json=batch)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert "batch_id" in body
    for pred in body["predictions"]:
        assert "churn_probability" in pred
        assert pred["prediction"] in (0, 1)


def test_health_endpoint(api_module):
    client = TestClient(api_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_endpoint(api_module):
    client = TestClient(api_module.app)
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus text format
    assert b"churn_api" in r.content
