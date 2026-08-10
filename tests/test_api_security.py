"""
Security regression tests for the HTTP API.

Each test here pins down a specific failure mode the service previously had:
authentication that silently disabled itself, endpoints that served model and
business internals to anonymous callers, and a batch size limit that only applied
after the whole request body had been parsed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

FAKE_METADATA = {
    "feature_schema": ["Tenure Months", "Monthly Charges", "Total Charges"],
    "feature_count": 3,
    "metrics": {},
    "feature_types": {
        "Tenure Months": "int",
        "Monthly Charges": "float",
        "Total Charges": "float",
    },
}

PAYLOAD = {"Tenure Months": 12, "Monthly Charges": 70.5, "Total Charges": 850.0}


def _patch_contract(monkeypatch):
    import churn_system.api.schema_generator as sg
    import churn_system.events.predictions as ep
    import churn_system.inference.model_contract as mc

    mc.load_model_contract.cache_clear()
    monkeypatch.setattr(mc, "load_model_contract", lambda: FAKE_METADATA)
    monkeypatch.setattr(sg, "load_model_contract", lambda: FAKE_METADATA)
    monkeypatch.setattr(ep, "load_model_contract", lambda: FAKE_METADATA)


def _stub_model():
    class StubModel:
        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))

    return StubModel()


@pytest.fixture
def secured_api(monkeypatch):
    """An API instance with authentication switched on."""
    monkeypatch.setenv("CHURN_DISABLE_RATE_LIMIT", "1")
    monkeypatch.setenv("CHURN_API_KEY", "prediction-key")
    monkeypatch.delenv("CHURN_ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("CHURN_ALLOW_ANONYMOUS", raising=False)
    _patch_contract(monkeypatch)

    import churn_system.api.api as api_mod

    importlib.reload(api_mod)
    monkeypatch.setattr(api_mod, "_get_model", lambda: _stub_model())
    return api_mod


class TestFailClosedAuth:
    """Authentication must not disable itself when misconfigured."""

    def test_import_refuses_when_no_key_and_no_explicit_opt_out(self, monkeypatch):
        """
        An unset CHURN_API_KEY previously made every ``Depends(verify_api_key)`` a
        no-op, and `docker compose up` with no .env was exactly that case. Refusing
        to start is the only safe behaviour.
        """
        # Import while the conftest opt-out is still in effect, so that the failure
        # under test comes from the reload and not from module import order.
        import churn_system.api.api as api_mod

        monkeypatch.delenv("CHURN_API_KEY", raising=False)
        monkeypatch.delenv("CHURN_ALLOW_ANONYMOUS", raising=False)
        _patch_contract(monkeypatch)

        with pytest.raises(RuntimeError, match="CHURN_API_KEY"):
            importlib.reload(api_mod)

        # A failed reload leaves the module half-initialised; restore it so later
        # tests are not affected by execution order.
        monkeypatch.setenv("CHURN_ALLOW_ANONYMOUS", "1")
        importlib.reload(api_mod)

    def test_anonymous_mode_requires_explicit_opt_in(self, monkeypatch):
        monkeypatch.delenv("CHURN_API_KEY", raising=False)
        monkeypatch.setenv("CHURN_ALLOW_ANONYMOUS", "1")
        _patch_contract(monkeypatch)

        import churn_system.api.api as api_mod

        importlib.reload(api_mod)  # must not raise
        assert api_mod.ALLOW_ANONYMOUS is True


class TestEnvFlagParsing:
    """`CHURN_DISABLE_RATE_LIMIT=0` used to disable rate limiting."""

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy_values_do_not_enable_the_flag(self, secured_api, monkeypatch, value):
        monkeypatch.setenv("CHURN_TEST_FLAG", value)
        assert secured_api._env_flag("CHURN_TEST_FLAG") is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_truthy_values_enable_the_flag(self, secured_api, monkeypatch, value):
        monkeypatch.setenv("CHURN_TEST_FLAG", value)
        assert secured_api._env_flag("CHURN_TEST_FLAG") is True


class TestEndpointAuthorization:
    """Endpoints that expose model or business internals must require a key."""

    @pytest.mark.parametrize("path", ["/monitoring/dashboard", "/explain/global"])
    def test_sensitive_endpoints_reject_anonymous_callers(self, secured_api, path):
        client = TestClient(secured_api.app)
        assert client.get(path).status_code == 401

    def test_dashboard_does_not_disclose_filesystem_paths(self, secured_api):
        client = TestClient(secured_api.app)
        response = client.get(
            "/monitoring/dashboard", headers={"X-API-Key": "prediction-key"}
        )

        assert response.status_code == 200
        assert "model_path" not in response.json()["model_info"]

    def test_health_and_metrics_stay_open(self, secured_api):
        """Probes must not require credentials — orchestrators do not have them."""
        client = TestClient(secured_api.app)
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

    def test_admin_endpoint_rejects_prediction_key_when_admin_key_set(
        self, secured_api, monkeypatch
    ):
        """A leaked prediction key must not be able to force model reloads."""
        monkeypatch.setenv("CHURN_ADMIN_API_KEY", "admin-key")
        importlib.reload(secured_api)

        client = TestClient(secured_api.app)
        response = client.post(
            "/admin/reload-model", headers={"X-API-Key": "prediction-key"}
        )

        assert response.status_code == 401


class TestRequestLimits:
    """Resource limits must apply before the work is done, not after."""

    def test_oversized_body_rejected_before_parsing(self, secured_api, monkeypatch):
        """
        The MAX_BATCH_SIZE check runs after FastAPI has already built every Pydantic
        model, so the body size has to be capped by middleware instead.
        """
        monkeypatch.setattr(secured_api, "MAX_BODY_BYTES", 1024)

        client = TestClient(secured_api.app)
        response = client.post(
            "/predict",
            content="x" * 4096,
            headers={"X-API-Key": "prediction-key", "Content-Type": "application/json"},
        )

        assert response.status_code == 413
        assert response.json()["error_code"] == "payload_too_large"

    def test_batch_over_limit_is_rejected(self, secured_api, monkeypatch):
        monkeypatch.setattr(secured_api, "MAX_BATCH_SIZE", 2)

        client = TestClient(secured_api.app)
        response = client.post(
            "/predict/batch",
            json=[PAYLOAD, PAYLOAD, PAYLOAD],
            headers={"X-API-Key": "prediction-key"},
        )

        assert response.status_code == 400
        # Flat envelope: every error status returns error_code at the top level.
        assert response.json()["error_code"] == "batch_too_large"

    def test_wrong_key_is_rejected(self, secured_api):
        client = TestClient(secured_api.app)
        response = client.post(
            "/predict", json=PAYLOAD, headers={"X-API-Key": "wrong-key"}
        )

        assert response.status_code == 401


class TestErrorEnvelope:
    """Every error response must have the same shape."""

    @pytest.mark.parametrize(
        ("method", "path", "kwargs", "status"),
        [
            ("post", "/predict", {"json": {"bad": 1}}, 422),
            ("get", "/monitoring/dashboard", {}, 401),
            ("post", "/feedback/does-not-exist", {"json": {"label": 1}}, 404),
        ],
    )
    def test_errors_expose_error_code_at_top_level(
        self, secured_api, method, path, kwargs, status
    ):
        client = TestClient(secured_api.app)
        if status != 401:
            kwargs.setdefault("headers", {})["X-API-Key"] = "prediction-key"

        response = getattr(client, method)(path, **kwargs)

        assert response.status_code == status
        body = response.json()
        assert "error_code" in body, f"{path} returned {body}"
        assert "message" in body

    def test_validation_errors_do_not_echo_submitted_values(self, secured_api):
        """The 422 detail should name fields, not reflect the caller's payload."""
        client = TestClient(secured_api.app)
        response = client.post(
            "/predict",
            json={"Tenure Months": "not-a-number", "secret": "sensitive-value"},
            headers={"X-API-Key": "prediction-key"},
        )

        assert response.status_code == 422
        assert "sensitive-value" not in response.text
