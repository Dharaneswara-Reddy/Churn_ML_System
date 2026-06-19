"""Unit tests for model promotion and rollback."""

from __future__ import annotations

import json

import pytest

from churn_system.lifecycle.promote import promote_model, schemas_match
from churn_system.lifecycle.rollback import rollback_if_needed


@pytest.fixture
def model_dirs(tmp_path, monkeypatch):
    """Set up experiment and production directories for promotion tests."""
    from churn_system.config import config as cfg

    experiments_dir = tmp_path / "experiments"
    production_dir = tmp_path / "production"
    production_dir.mkdir(parents=True)
    experiments_dir.mkdir(parents=True)

    monkeypatch.setitem(cfg.CONFIG["paths"], "experiments_dir", str(experiments_dir))
    monkeypatch.setitem(cfg.CONFIG["paths"], "production_model", str(production_dir / "model.pkl"))

    return experiments_dir, production_dir


@pytest.fixture
def experiment_v1(model_dirs):
    """Create a valid experiment version with model + metadata."""
    experiments_dir, _ = model_dirs
    exp_dir = experiments_dir / "churn_model_20260301_120000"
    exp_dir.mkdir()

    # Create model.pkl
    (exp_dir / "model.pkl").write_bytes(b"fake_model_data")

    # Create metadata
    metadata = {
        "model_version": "20260301_120000",
        "feature_schema": ["A", "B", "C"],
        "metrics": {"roc_auc": 0.85, "pr_auc": 0.60},
        "dataset": "data/test.csv",
    }
    (exp_dir / "metadata.json").write_text(json.dumps(metadata))

    return exp_dir


class TestSchemasMatch:
    """Test schema matching logic used during promotion."""

    def test_identical_schemas_match(self):
        prod = {"feature_schema": ["A", "B", "C"]}
        new = {"feature_schema": ["A", "B", "C"]}
        assert schemas_match(prod, new) is True

    def test_different_schemas_dont_match(self):
        prod = {"feature_schema": ["A", "B", "C"]}
        new = {"feature_schema": ["A", "B"]}
        assert schemas_match(prod, new) is False

    def test_empty_schemas_match(self):
        prod = {"feature_schema": []}
        new = {"feature_schema": []}
        assert schemas_match(prod, new) is True

    def test_missing_schema_key(self):
        prod = {}
        new = {"feature_schema": ["A"]}
        assert schemas_match(prod, new) is False


class TestPromoteModel:
    """Test model promotion workflow."""

    def test_promote_creates_production_copy(self, model_dirs, experiment_v1, monkeypatch):
        experiments_dir, production_dir = model_dirs

        # Patch lineage to avoid file I/O side effects
        monkeypatch.setattr(
            "churn_system.lifecycle.promote.record_lineage",
            lambda **kwargs: None,
        )

        promote_model("churn_model_20260301_120000")

        # promote_model copies into production_dir / "current"
        promoted_dir = production_dir / "current"
        assert (promoted_dir / "model.pkl").exists()
        assert (promoted_dir / "metadata.json").exists()

    def test_promote_nonexistent_version_raises(self, model_dirs):
        with pytest.raises(ValueError, match="does not exist"):
            promote_model("churn_model_99999999_000000")


class TestRollback:
    """Test rollback logic."""

    def test_rollback_skipped_when_no_health_report(self, tmp_path, monkeypatch):
        from churn_system.config import config as cfg

        monkeypatch.setitem(cfg.CONFIG["paths"], "monitoring_dir", str(tmp_path / "monitoring"))
        monkeypatch.setitem(cfg.CONFIG["paths"], "lineage_path", str(tmp_path / "lineage.json"))

        # Should not raise — just skip
        rollback_if_needed()

    def test_rollback_skipped_when_model_healthy(self, tmp_path, monkeypatch):
        from churn_system.config import config as cfg

        monitoring_dir = tmp_path / "monitoring"
        monitoring_dir.mkdir()
        health_file = monitoring_dir / "health_report.json"
        health_file.write_text(json.dumps({"retraining_recommended": False}))

        monkeypatch.setitem(cfg.CONFIG["paths"], "monitoring_dir", str(monitoring_dir))
        monkeypatch.setitem(cfg.CONFIG["paths"], "lineage_path", str(tmp_path / "lineage.json"))

        rollback_if_needed()
