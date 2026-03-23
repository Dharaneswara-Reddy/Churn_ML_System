"""Regression: /predict happy path with mocked model."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_module(monkeypatch):
    monkeypatch.setenv("CHURN_DISABLE_RATE_LIMIT", "1")
    monkeypatch.delenv("CHURN_API_KEY", raising=False)
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


def test_predict_returns_probability(api_module, monkeypatch):
    class StubModel:
        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))

    monkeypatch.setattr(api_module, "get_model", lambda: StubModel())

    client = TestClient(api_module.app)
    r = client.post("/predict", json=_sample_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert "churn_probability" in body
    assert "request_id" in body
    assert body["prediction"] in (0, 1)


def test_predict_rejects_extra_field(api_module, monkeypatch):
    class StubModel:
        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

    monkeypatch.setattr(api_module, "get_model", lambda: StubModel())

    p = _sample_payload()
    p["unexpected"] = 1
    client = TestClient(api_module.app)
    r = client.post("/predict", json=p)
    assert r.status_code == 422


def test_predict_requires_api_key_when_set(monkeypatch):
    monkeypatch.setenv("CHURN_DISABLE_RATE_LIMIT", "1")
    monkeypatch.setenv("CHURN_API_KEY", "secret")
    import churn_system.api.api as api_mod

    importlib.reload(api_mod)

    class StubModel:
        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))

    monkeypatch.setattr(api_mod, "get_model", lambda: StubModel())

    client = TestClient(api_mod.app)
    r = client.post("/predict", json=_sample_payload())
    assert r.status_code == 401

    r2 = client.post(
        "/predict",
        json=_sample_payload(),
        headers={"X-API-Key": "secret"},
    )
    assert r2.status_code == 200
