"""
Model artifact integrity tests.

``pickle.load`` is arbitrary code execution by design. In this deployment the
training and scheduler containers mount ``./models`` read-write while the API
mounts it read-only, and a promotion automatically triggers a hot reload — so a
compromised training job could previously write a malicious pickle that the API
would execute on its next reload, with no human in the loop and nothing to detect
it. ``validate_model_bundle`` checked only that ``metadata.json`` was well-formed
JSON, which is not an integrity control.

These tests pin down that verification happens, that it happens *before*
deserialisation, and that it fails closed.
"""

from __future__ import annotations

import json
import pickle

import pytest

from churn_system.artifacts import (
    ArtifactSignatureError,
    sign_model_bundle,
    verify_bundle_signature,
)


@pytest.fixture
def signed_bundle(tmp_path, monkeypatch):
    """A correctly signed bundle on disk."""
    monkeypatch.setenv("CHURN_ARTIFACT_SIGNING_KEY", "unit-test-key")
    monkeypatch.delenv("CHURN_ALLOW_UNSIGNED_ARTIFACTS", raising=False)

    bundle = tmp_path / "current"
    bundle.mkdir()
    (bundle / "model.pkl").write_bytes(pickle.dumps({"model": "legitimate"}))
    (bundle / "metadata.json").write_text(
        json.dumps({"model_version": "v1", "feature_schema": ["A", "B"]})
    )
    sign_model_bundle(bundle)
    return bundle


class TestSignatureVerification:
    def test_valid_signed_bundle_verifies(self, signed_bundle):
        verify_bundle_signature(signed_bundle)  # must not raise

    def test_tampered_model_is_rejected(self, signed_bundle):
        """The core attack: swap the pickle, keep everything else intact."""
        (signed_bundle / "model.pkl").write_bytes(
            pickle.dumps({"model": "malicious payload"})
        )

        with pytest.raises(ArtifactSignatureError):
            verify_bundle_signature(signed_bundle)

    def test_tampered_metadata_is_rejected(self, signed_bundle):
        """Metadata is signed too — the feature schema drives serving behaviour."""
        (signed_bundle / "metadata.json").write_text(
            json.dumps({"model_version": "v1", "feature_schema": ["A", "B", "EVIL"]})
        )

        with pytest.raises(ArtifactSignatureError):
            verify_bundle_signature(signed_bundle)

    def test_missing_signature_is_rejected(self, signed_bundle):
        (signed_bundle / "signature.json").unlink()

        with pytest.raises(ArtifactSignatureError):
            verify_bundle_signature(signed_bundle)

    def test_signature_from_a_different_key_is_rejected(self, signed_bundle, monkeypatch):
        """An attacker who can write the bundle still cannot forge the signature."""
        monkeypatch.setenv("CHURN_ARTIFACT_SIGNING_KEY", "attacker-key")

        with pytest.raises(ArtifactSignatureError):
            verify_bundle_signature(signed_bundle)

    def test_corrupt_signature_file_is_rejected(self, signed_bundle):
        (signed_bundle / "signature.json").write_text("not json at all")

        with pytest.raises(ArtifactSignatureError):
            verify_bundle_signature(signed_bundle)


class TestFailClosed:
    def test_unsigned_bundle_refused_when_no_key_configured(self, tmp_path, monkeypatch):
        """
        Absence of a signing key must refuse, not silently pass. A verification
        step that no-ops when unconfigured provides no security at all.
        """
        monkeypatch.delenv("CHURN_ARTIFACT_SIGNING_KEY", raising=False)
        monkeypatch.delenv("CHURN_ALLOW_UNSIGNED_ARTIFACTS", raising=False)

        bundle = tmp_path / "current"
        bundle.mkdir()
        (bundle / "model.pkl").write_bytes(b"anything")
        (bundle / "metadata.json").write_text(json.dumps({"feature_schema": ["A"]}))

        with pytest.raises(ArtifactSignatureError):
            verify_bundle_signature(bundle)

    def test_explicit_opt_out_is_honoured(self, tmp_path, monkeypatch):
        """The escape hatch exists, but must be deliberate and explicit."""
        monkeypatch.delenv("CHURN_ARTIFACT_SIGNING_KEY", raising=False)
        monkeypatch.setenv("CHURN_ALLOW_UNSIGNED_ARTIFACTS", "1")

        bundle = tmp_path / "current"
        bundle.mkdir()
        (bundle / "model.pkl").write_bytes(b"anything")
        (bundle / "metadata.json").write_text(json.dumps({"feature_schema": ["A"]}))

        verify_bundle_signature(bundle)  # must not raise

    def test_opt_out_does_not_excuse_a_bad_signature(self, signed_bundle, monkeypatch):
        """
        Once a key IS configured, the opt-out must not rescue a tampered bundle —
        otherwise an attacker who can set one env var disables integrity entirely.
        """
        monkeypatch.setenv("CHURN_ALLOW_UNSIGNED_ARTIFACTS", "1")
        (signed_bundle / "model.pkl").write_bytes(pickle.dumps({"model": "malicious"}))

        with pytest.raises(ArtifactSignatureError):
            verify_bundle_signature(signed_bundle)


class TestServingPathIsGated:
    """
    The registry must verify before unpickling — not merely somewhere in the
    codebase. The audit found signing implemented in artifacts.py while
    ``_load_model_from_disk`` still did a raw ``pickle.load`` that bypassed it.
    """

    def test_registry_refuses_a_tampered_bundle(self, signed_bundle, monkeypatch):
        from churn_system.config import config as cfg
        from churn_system.serving.model_registry import ModelRegistry

        monkeypatch.setitem(
            cfg.CONFIG["paths"], "production_model", str(signed_bundle / "model.pkl")
        )
        (signed_bundle / "model.pkl").write_bytes(pickle.dumps({"model": "malicious"}))

        ModelRegistry.reset()
        with pytest.raises(ArtifactSignatureError):
            ModelRegistry.instance().get_bundle()

    def test_registry_loads_a_valid_bundle(self, signed_bundle, monkeypatch):
        from churn_system.config import config as cfg
        from churn_system.serving.model_registry import ModelRegistry

        monkeypatch.setitem(
            cfg.CONFIG["paths"], "production_model", str(signed_bundle / "model.pkl")
        )

        ModelRegistry.reset()
        bundle = ModelRegistry.instance().get_bundle()

        assert bundle.model == {"model": "legitimate"}
        assert bundle.version == "v1"
        assert bundle.feature_schema == ("A", "B")

    def test_verification_precedes_deserialisation(self, signed_bundle, monkeypatch):
        """
        Order matters: verifying *after* unpickling would be useless, since the
        payload has already executed by then.
        """
        from churn_system.config import config as cfg
        from churn_system.serving import model_registry as registry_module

        monkeypatch.setitem(
            cfg.CONFIG["paths"], "production_model", str(signed_bundle / "model.pkl")
        )

        calls: list[str] = []

        def spy_verify(bundle_dir):
            calls.append("verify")

        def spy_load(handle):
            calls.append("unpickle")
            return {"model": "legitimate"}

        monkeypatch.setattr(registry_module, "verify_bundle_signature", spy_verify)
        monkeypatch.setattr(registry_module.pickle, "load", spy_load)

        registry_module.ModelRegistry.reset()
        registry_module.ModelRegistry.instance().get_bundle()

        assert calls == ["verify", "unpickle"], calls
