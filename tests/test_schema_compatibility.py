"""
Backward compatibility of the dynamically generated request schema.

The API's request model is derived from the champion's ``feature_schema``. That is
a good property — the schema can never drift from the model — but it has a sharp
edge: promoting a champion with fewer features silently deletes fields from the
public API. With ``extra="forbid"``, every previously valid request becomes a 422
the moment a *background scheduler* promotes a model. A breaking API change
delivered by a cron job, with no deploy and no version bump.

Removing the seven geographic features is exactly that situation. These tests pin
the compatibility contract that lets it ship without breaking existing callers.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from churn_system.features.build_features import GEOGRAPHIC_COLUMNS

# The 19 features the geography-free champion actually consumes.
_ACTIVE_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
]

_FEATURE_TYPES = {
    **dict.fromkeys(_ACTIVE_FEATURES, "str"),
    "Tenure Months": "int",
    "Monthly Charges": "float",
    "Total Charges": "float",
}

class ConstantModel:
    """
    A stand-in estimator with just enough of the interface for the serving path.

    Defined at module scope on purpose: ``pickle`` stores a class by qualified
    name, so a class defined inside a fixture cannot be pickled at all.
    """

    def predict_proba(self, frame):
        import numpy as np

        return np.column_stack([np.full(len(frame), 0.7), np.full(len(frame), 0.3)])


_NEW_CLIENT_PAYLOAD = {
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 70.35,
    "Total Charges": 845.5,
}

# What a client written against the previous 26-feature champion sends.
_OLD_CLIENT_PAYLOAD = {
    **_NEW_CLIENT_PAYLOAD,
    "Country": "United States",
    "State": "California",
    "City": "Los Angeles",
    "Zip Code": "90003",
    "Lat Long": "33.964131, -118.272783",
    "Latitude": 33.964131,
    "Longitude": -118.272783,
}


@pytest.fixture
def geo_free_contract(monkeypatch):
    """Pretend the geography-free model is the champion."""
    import churn_system.api.schema_generator as gen
    import churn_system.inference.model_contract as contract

    metadata = {
        "model_version": "geo_free_v1",
        "feature_schema": list(_ACTIVE_FEATURES),
        "feature_types": dict(_FEATURE_TYPES),
        "operating_threshold": 0.14,
        "metrics": {"roc_auc": 0.8534, "pr_auc": 0.6733},
    }

    contract.clear_model_contract_cache()
    monkeypatch.setattr(gen, "load_model_contract", lambda: metadata)
    monkeypatch.setattr(contract, "load_model_contract", lambda: metadata)
    return metadata


class TestDeprecatedFieldSet:
    def test_removed_geography_becomes_deprecated_not_forbidden(self, geo_free_contract):
        from churn_system.api.schema_generator import deprecated_request_fields

        deprecated = set(deprecated_request_fields())

        assert set(GEOGRAPHIC_COLUMNS) <= deprecated

    def test_active_features_are_never_deprecated(self, geo_free_contract):
        from churn_system.api.schema_generator import deprecated_request_fields

        deprecated = set(deprecated_request_fields())

        assert deprecated.isdisjoint(_ACTIVE_FEATURES)

    def test_the_target_column_is_never_accepted(self, geo_free_contract):
        """
        Sending ``Churn Value`` to a prediction endpoint is a caller bug — probably
        a training frame posted by mistake — and must stay a 422 rather than being
        absorbed as legacy noise.
        """
        from churn_system.api.schema_generator import deprecated_request_fields

        assert "Churn Value" not in deprecated_request_fields()

    def test_leakage_columns_are_never_accepted(self, geo_free_contract):
        """
        ``build_features`` drops leakage columns too, but they were never fields of
        any published request schema — no client has them to send. Accepting them
        would widen the public API to take ``Churn Label`` and ``Churn Score``, the
        target restated, and silently answer instead of telling the caller they are
        posting training data to an inference endpoint.
        """
        from churn_system.api.schema_generator import deprecated_request_fields
        from churn_system.features.build_features import LEAKAGE_COLUMNS

        deprecated = set(deprecated_request_fields())

        assert deprecated.isdisjoint(LEAKAGE_COLUMNS)
        assert "CustomerID" not in deprecated

    def test_only_geography_is_deprecated(self, geo_free_contract):
        from churn_system.api.schema_generator import deprecated_request_fields

        assert set(deprecated_request_fields()) == set(GEOGRAPHIC_COLUMNS)

    def test_the_shim_retires_itself_when_a_feature_returns(self, monkeypatch):
        """
        The deprecated set is derived from the champion, not hardcoded. A model that
        starts using ``City`` again must make ``City`` a required field, not leave
        it silently ignored — otherwise the API would accept and discard a value
        the model needs.
        """
        import churn_system.api.schema_generator as gen

        monkeypatch.setattr(
            gen,
            "load_model_contract",
            lambda: {
                "model_version": "with_city",
                "feature_schema": [*_ACTIVE_FEATURES, "City"],
                "feature_types": {**_FEATURE_TYPES, "City": "str"},
            },
        )

        assert "City" not in gen.deprecated_request_fields()

    def test_strict_mode_disables_the_shim_entirely(self, geo_free_contract, monkeypatch):
        from churn_system.api.schema_generator import deprecated_request_fields

        monkeypatch.setenv("CHURN_STRICT_REQUEST_SCHEMA", "1")

        assert deprecated_request_fields() == []


class TestGeneratedRequestModel:
    def test_an_old_client_payload_still_validates(self, geo_free_contract):
        from churn_system.api.schema_generator import generate_request_model

        model = generate_request_model()

        instance = model(**_OLD_CLIENT_PAYLOAD)

        assert instance.model_dump()["Tenure Months"] == 12

    def test_a_new_client_may_omit_the_deprecated_fields(self, geo_free_contract):
        from churn_system.api.schema_generator import generate_request_model

        model = generate_request_model()

        instance = model(**_NEW_CLIENT_PAYLOAD)

        assert instance.model_dump()["City"] is None

    def test_an_unknown_field_is_still_rejected(self, geo_free_contract):
        """
        The shim must not become a blanket ``extra="allow"``. A misspelled feature
        name has to stay a hard error, or a client can silently send
        ``"Tenure_Months"`` forever and get predictions from a default-filled row.
        """
        from pydantic import ValidationError

        from churn_system.api.schema_generator import generate_request_model

        model = generate_request_model()

        with pytest.raises(ValidationError):
            model(**{**_NEW_CLIENT_PAYLOAD, "Tenure_Months": 12})

    def test_a_missing_active_feature_is_still_rejected(self, geo_free_contract):
        from pydantic import ValidationError

        from churn_system.api.schema_generator import generate_request_model

        payload = dict(_NEW_CLIENT_PAYLOAD)
        payload.pop("Contract")
        model = generate_request_model()

        with pytest.raises(ValidationError):
            model(**payload)

    def test_deprecated_fields_are_marked_deprecated_in_the_schema(self, geo_free_contract):
        """
        Callers need a machine-readable signal, not just a changelog entry. OpenAPI
        renders ``deprecated: true`` in generated clients and docs.
        """
        from churn_system.api.schema_generator import generate_request_model

        schema = generate_request_model().model_json_schema()

        assert schema["properties"]["City"].get("deprecated") is True
        assert "deprecated" not in schema["properties"]["Contract"]

    def test_strict_mode_rejects_the_old_payload(self, geo_free_contract, monkeypatch):
        from pydantic import ValidationError

        from churn_system.api.schema_generator import generate_request_model

        monkeypatch.setenv("CHURN_STRICT_REQUEST_SCHEMA", "1")
        model = generate_request_model()

        with pytest.raises(ValidationError):
            model(**_OLD_CLIENT_PAYLOAD)


class TestDeprecatedFieldsNeverReachTheModelOrTheStore:
    """
    Accepting geography is not the same as collecting it.

    The whole reason geography was removed is that it was memorised by the model
    and written into the durable event store. A compatibility shim that quietly
    forwarded those values would reintroduce exactly the problem it was covering
    for.
    """

    def test_the_row_handed_to_inference_has_no_deprecated_keys(self, geo_free_contract):
        import churn_system.api.api as api_mod
        from churn_system.api.schema_generator import generate_request_model

        payload = generate_request_model()(**_OLD_CLIENT_PAYLOAD)

        row = api_mod._payload_to_row(payload)

        for column in GEOGRAPHIC_COLUMNS:
            assert column not in row, f"{column} survived into the inference row"
        assert row["Tenure Months"] == 12

    def test_omitted_deprecated_fields_are_not_added_as_none(self, geo_free_contract):
        """
        A ``None`` for ``City`` would still be a column reaching ``build_features``
        and, on a strict Pandera schema, a validation failure for the new caller
        who did everything right.
        """
        import churn_system.api.api as api_mod
        from churn_system.api.schema_generator import generate_request_model

        payload = generate_request_model()(**_NEW_CLIENT_PAYLOAD)

        row = api_mod._payload_to_row(payload)

        assert set(row) == set(_ACTIVE_FEATURES)

    def test_deprecated_field_usage_is_counted(self, geo_free_contract):
        """
        Retiring the shim needs evidence that no caller still sends the fields.
        Without a counter that question can only be answered by guessing.
        """
        import churn_system.api.api as api_mod
        from churn_system.api.schema_generator import generate_request_model
        from churn_system.observability.metrics import DEPRECATED_REQUEST_FIELDS_TOTAL

        before = DEPRECATED_REQUEST_FIELDS_TOTAL.labels(field="City")._value.get()

        api_mod._payload_to_row(generate_request_model()(**_OLD_CLIENT_PAYLOAD))

        after = DEPRECATED_REQUEST_FIELDS_TOTAL.labels(field="City")._value.get()
        assert after == before + 1

    def test_omitting_a_deprecated_field_does_not_count_it(self, geo_free_contract):
        import churn_system.api.api as api_mod
        from churn_system.api.schema_generator import generate_request_model
        from churn_system.observability.metrics import DEPRECATED_REQUEST_FIELDS_TOTAL

        before = DEPRECATED_REQUEST_FIELDS_TOTAL.labels(field="Latitude")._value.get()

        api_mod._payload_to_row(generate_request_model()(**_NEW_CLIENT_PAYLOAD))

        after = DEPRECATED_REQUEST_FIELDS_TOTAL.labels(field="Latitude")._value.get()
        assert after == before


class TestLiveApiAcceptsBothClients:
    """
    End to end through the real FastAPI app, against a real signed bundle.

    The bundle is built inside ``tmp_path`` rather than reusing
    ``models/production/current``: a test that reads the deployed bundle passes or
    fails depending on which model happens to be promoted and which signing key the
    developer has exported, which is exactly the kind of order- and
    environment-dependence the rest of this suite works to avoid.
    """

    @pytest.fixture
    def deployed(self, isolated_paths, monkeypatch):
        import importlib
        import pickle

        from churn_system.artifacts import sign_model_bundle
        from churn_system.inference import model_contract

        bundle = isolated_paths["production_model"].parent
        bundle.mkdir(parents=True, exist_ok=True)

        with open(bundle / "model.pkl", "wb") as handle:
            pickle.dump(ConstantModel(), handle)
        (bundle / "metadata.json").write_text(
            json.dumps(
                {
                    "model_version": "geo_free_v1",
                    "feature_schema": list(_ACTIVE_FEATURES),
                    "feature_types": dict(_FEATURE_TYPES),
                    "operating_threshold": 0.14,
                    "metrics": {"roc_auc": 0.8534, "pr_auc": 0.6733},
                }
            )
        )
        sign_model_bundle(bundle)

        model_contract.clear_model_contract_cache()

        import churn_system.api.api as api_mod

        importlib.reload(api_mod)
        try:
            yield TestClient(api_mod.app), api_mod
        finally:
            model_contract.clear_model_contract_cache()
            importlib.reload(api_mod)

    def test_openapi_documents_the_deprecated_fields(self, deployed):
        _, api_mod = deployed

        schema = api_mod.RequestModel.model_json_schema()

        for field in GEOGRAPHIC_COLUMNS:
            assert schema["properties"][field].get("deprecated") is True
            assert field not in schema.get("required", [])

    def test_the_openapi_document_is_still_generated(self, deployed):
        client, _ = deployed

        response = client.get("/openapi.json")

        assert response.status_code == 200
        assert "DynamicPredictionRequest" in json.dumps(response.json())

    def test_an_old_client_gets_a_prediction_not_a_422(self, deployed):
        """
        The headline guarantee. Before the shim, promoting the geography-free model
        turned every request from an existing caller into a 422 — shipped by the
        scheduler, with no deploy.
        """
        client, _ = deployed

        response = client.post("/predict", json=_OLD_CLIENT_PAYLOAD)

        assert response.status_code == 200, response.text
        assert "churn_probability" in response.json()

    def test_both_clients_receive_the_same_prediction(self, deployed):
        """
        Accepting the geographic fields must be indistinguishable from omitting
        them. If the two answers differed, the fields would still be influencing
        the model — which is the thing removing them was meant to stop.
        """
        client, _ = deployed

        old = client.post("/predict", json=_OLD_CLIENT_PAYLOAD).json()
        new = client.post("/predict", json=_NEW_CLIENT_PAYLOAD).json()

        assert old["churn_probability"] == new["churn_probability"]

    def test_the_operating_threshold_comes_from_the_bundle(self, deployed):
        client, _ = deployed

        assert client.post("/predict", json=_NEW_CLIENT_PAYLOAD).json()["threshold"] == 0.14

    def test_a_misspelled_field_is_still_a_422(self, deployed):
        client, _ = deployed

        response = client.post(
            "/predict", json={**_NEW_CLIENT_PAYLOAD, "Tenure_Months": 12}
        )

        assert response.status_code == 422

    def test_batch_predictions_accept_a_mixed_fleet_of_clients(self, deployed):
        """
        A rolling client upgrade means old and new payloads arrive in the same
        batch. Both must be accepted in one request.
        """
        client, _ = deployed

        response = client.post(
            "/predict/batch", json=[_OLD_CLIENT_PAYLOAD, _NEW_CLIENT_PAYLOAD]
        )

        assert response.status_code == 200, response.text
        assert len(response.json()["predictions"]) == 2
