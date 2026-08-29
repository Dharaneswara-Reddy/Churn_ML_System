"""Unit tests for model promotion and rollback."""

from __future__ import annotations

import json

import pytest

from churn_system.lifecycle.model_compare import compare_models
from churn_system.lifecycle.promote import promote_model, schemas_match
from churn_system.lifecycle.rollback import rollback_if_needed


def _write_bundle(directory, *, schema, metrics, version="20260101_000000"):
    """Create a minimal model bundle on disk."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.pkl").write_bytes(b"model")
    (directory / "metadata.json").write_text(
        json.dumps(
            {"model_version": version, "feature_schema": schema, "metrics": metrics}
        )
    )
    return directory


@pytest.fixture
def model_dirs(tmp_path, monkeypatch):
    """Set up experiment and production directories for promotion tests."""
    from churn_system.config import config as cfg

    experiments_dir = tmp_path / "experiments"
    production_dir = tmp_path / "production"
    production_dir.mkdir(parents=True)
    experiments_dir.mkdir(parents=True)

    monkeypatch.setitem(cfg.CONFIG["paths"], "experiments_dir", str(experiments_dir))
    monkeypatch.setitem(cfg.CONFIG["paths"], "production_model", str(production_dir / "current" / "model.pkl"))

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
        _, production_dir = model_dirs

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


    def test_promotion_blocked_on_schema_mismatch(self, model_dirs, experiment_v1):
        """
        The schema interlock must refuse the promotion and leave production intact.

        Serving selects and orders inference columns from the production
        metadata.json, so promoting a bundle with a different feature schema breaks
        every prediction. This guard previously never executed under test.
        """
        _, production_dir = model_dirs
        current = production_dir / "current"
        current.mkdir(parents=True)
        (current / "model.pkl").write_bytes(b"incumbent_model")
        (current / "metadata.json").write_text(
            json.dumps(
                {
                    "model_version": "20260101_000000",
                    "feature_schema": ["A", "B"],  # challenger has ["A", "B", "C"]
                    "metrics": {"roc_auc": 0.80},
                }
            )
        )

        promoted = promote_model("churn_model_20260301_120000")

        assert promoted is False, "mismatched schema must not be promoted"
        assert (current / "model.pkl").read_bytes() == b"incumbent_model"
        assert json.loads((current / "metadata.json").read_text())["feature_schema"] == [
            "A",
            "B",
        ]

    def test_promotion_reports_success(self, model_dirs, experiment_v1, monkeypatch):
        """Callers must be able to tell a promotion from a refusal."""
        monkeypatch.setattr(
            "churn_system.lifecycle.promote.record_lineage",
            lambda **kwargs: None,
        )

        assert promote_model("churn_model_20260301_120000") is True

    def test_promotion_leaves_no_staging_directories(
        self, model_dirs, experiment_v1, monkeypatch
    ):
        """The atomic swap must clean up after itself."""
        _, production_dir = model_dirs
        monkeypatch.setattr(
            "churn_system.lifecycle.promote.record_lineage",
            lambda **kwargs: None,
        )

        promote_model("churn_model_20260301_120000")

        # Staging artifacts must be cleaned up. The flock sentinel (.current.lock)
        # is deliberately excluded: it is the lock file itself and must persist for
        # mutual exclusion between concurrent promotions, so its presence is correct.
        leftovers = [
            entry.name
            for entry in production_dir.iterdir()
            if entry.is_dir() and ("incoming" in entry.name or "retired" in entry.name)
        ]
        assert leftovers == [], f"staging directories left behind: {leftovers}"

        # The sentinel must be a file, not a stray directory.
        lock_files = [e for e in production_dir.iterdir() if e.name.endswith(".lock")]
        assert all(e.is_file() for e in lock_files)


class TestCompareModels:
    """The promotion gate must honour the configured metric and margin."""

    @pytest.fixture
    def champion_and_challenger(self, isolated_paths):
        """
        Build a champion/challenger pair.

        Metrics are padded with recall/precision that clear the absolute floors,
        and an experiment report carrying a bootstrap interval, so these tests
        exercise the metric-selection logic specifically rather than tripping an
        unrelated gate. Tests that target a particular gate override these.
        """

        def build(champion_metrics, challenger_metrics, ci_lower=None):
            champion = {"recall": 0.60, "precision": 0.45, **champion_metrics}
            challenger = {"recall": 0.62, "precision": 0.46, **challenger_metrics}

            _write_bundle(
                isolated_paths["production_model"].parent,
                schema=["A", "B"],
                metrics=champion,
            )
            experiment = isolated_paths["experiments_dir"] / "churn_model_20260301_120000"
            _write_bundle(experiment, schema=["A", "B"], metrics=challenger)

            metric = "pr_auc"
            if ci_lower is None:
                # Comfortably above the champion, so significance is not the gate
                # under test here.
                ci_lower = float(champion.get(metric, 0.0)) + 0.05
            (experiment / "experiment_report.json").write_text(
                json.dumps(
                    {
                        "winner": "candidate",
                        "selection_metric": metric,
                        "confidence_intervals": {
                            "candidate": {"lower": ci_lower, "upper": 1.0}
                        },
                    }
                )
            )

        return build

    def test_uses_configured_metric_not_roc_auc(
        self, champion_and_challenger, monkeypatch
    ):
        """
        A challenger that improves PR-AUC but dips slightly on ROC-AUC must win.

        Training selects its winner by PR-AUC (the target is imbalanced), so judging
        promotion by ROC-AUC could reject the model training just chose.
        """
        from churn_system.config import config as cfg

        monkeypatch.setitem(cfg.CONFIG["model_promotion"], "metric", "pr_auc")
        # pr_auc floor lowered so this test isolates *which metric* is compared,
        # not whether the absolute value clears the production floor.
        monkeypatch.setitem(cfg.CONFIG["model_promotion"], "min_pr_auc", 0.0)
        champion_and_challenger(
            {"roc_auc": 0.8225, "pr_auc": 0.22},
            {"roc_auc": 0.8220, "pr_auc": 0.31},
        )

        assert compare_models() is True

    def test_respects_min_improvement_margin(
        self, champion_and_challenger, monkeypatch
    ):
        """A negligible gain must not trigger a production model swap."""
        from churn_system.config import config as cfg

        monkeypatch.setitem(cfg.CONFIG["model_promotion"], "metric", "pr_auc")
        monkeypatch.setitem(cfg.CONFIG["model_promotion"], "min_improvement", 0.01)
        champion_and_challenger({"pr_auc": 0.30}, {"pr_auc": 0.3000001})
        # Only the improvement gate should decide this case.
        monkeypatch.setitem(cfg.CONFIG["model_promotion"], "min_pr_auc", 0.0)

        assert compare_models() is False

    def test_refuses_when_metric_missing(self, champion_and_challenger, monkeypatch):
        """
        Missing metrics previously defaulted to 0, so any challenger beat a champion
        whose metadata lacked the key.
        """
        from churn_system.config import config as cfg

        monkeypatch.setitem(cfg.CONFIG["model_promotion"], "metric", "pr_auc")
        champion_and_challenger({"roc_auc": 0.9}, {"pr_auc": 0.05})

        assert compare_models() is False

    def test_promotes_first_model_when_no_champion(self, isolated_paths):
        _write_bundle(
            isolated_paths["experiments_dir"] / "churn_model_20260301_120000",
            schema=["A", "B"],
            metrics={"pr_auc": 0.2},
        )

        assert compare_models() is True


class TestRollback:
    """Test rollback logic."""

    def test_rollback_skipped_when_no_health_report(self, tmp_path, monkeypatch):
        from churn_system.config import config as cfg

        monkeypatch.setitem(cfg.CONFIG["paths"], "monitoring_dir", str(tmp_path / "monitoring"))
        monkeypatch.setitem(cfg.CONFIG["paths"], "lineage_path", str(tmp_path / "lineage.json"))

        # Should not raise — just skip
        assert rollback_if_needed() is False

    def test_rollback_skipped_when_model_healthy(self, tmp_path, monkeypatch):
        from churn_system.config import config as cfg

        monitoring_dir = tmp_path / "monitoring"
        monitoring_dir.mkdir()
        health_file = monitoring_dir / "health_report.json"
        health_file.write_text(json.dumps({"retraining_recommended": False}))

        monkeypatch.setitem(cfg.CONFIG["paths"], "monitoring_dir", str(monitoring_dir))
        monkeypatch.setitem(cfg.CONFIG["paths"], "lineage_path", str(tmp_path / "lineage.json"))

        assert rollback_if_needed() is False

    def test_rollback_does_not_touch_paths_outside_the_test(self, isolated_paths):
        """
        Guard against the whole class of bug this test file used to have.

        ``rollback.py`` bound its paths at import, so monkeypatching CONFIG afterwards
        had no effect and the "skipped" cases actually deleted and recopied the real
        models/production/current directory on every run.
        """
        from churn_system.lifecycle import rollback

        assert rollback._health_path().is_relative_to(isolated_paths["monitoring_dir"].parent)
        assert rollback._lineage_path() == isolated_paths["lineage_path"]

    def test_rollback_restores_previous_model(self, isolated_paths):
        """The actual restore path — previously never executed by any test."""
        experiments = isolated_paths["experiments_dir"]
        production = isolated_paths["production_model"].parent

        for version, payload in [
            ("churn_model_20260101_000000", b"old_model"),
            ("churn_model_20260201_000000", b"new_model"),
        ]:
            exp = experiments / version
            exp.mkdir(parents=True)
            (exp / "model.pkl").write_bytes(payload)
            (exp / "metadata.json").write_text(json.dumps({"model_version": version}))

        production.mkdir(parents=True)
        (production / "model.pkl").write_bytes(b"new_model")

        isolated_paths["lineage_path"].parent.mkdir(parents=True, exist_ok=True)
        isolated_paths["lineage_path"].write_text(
            json.dumps(
                [
                    {"model_version": "churn_model_20260101_000000"},
                    {"model_version": "churn_model_20260201_000000"},
                ]
            )
        )
        (isolated_paths["monitoring_dir"] / "health_report.json").write_text(
            json.dumps({"retraining_recommended": True})
        )

        assert rollback_if_needed() is True
        assert (production / "model.pkl").read_bytes() == b"old_model"

    def test_rollback_refuses_when_lineage_has_one_distinct_version(
        self, isolated_paths
    ):
        """
        Repeated promotions of the same version must not count as a rollback target.

        Lineage records every promotion, so ``lineage[-2]`` is often the version
        already in production — "restoring" it is a no-op dressed up as recovery.
        """
        production = isolated_paths["production_model"].parent
        production.mkdir(parents=True)
        (production / "model.pkl").write_bytes(b"current_model")

        isolated_paths["lineage_path"].parent.mkdir(parents=True, exist_ok=True)
        isolated_paths["lineage_path"].write_text(
            json.dumps(
                [
                    {"model_version": "churn_model_20260201_000000"},
                    {"model_version": "churn_model_20260201_000000"},
                ]
            )
        )
        (isolated_paths["monitoring_dir"] / "health_report.json").write_text(
            json.dumps({"retraining_recommended": True})
        )

        assert rollback_if_needed() is False
        assert (production / "model.pkl").read_bytes() == b"current_model"


class TestRollbackSafety:
    """Rollback is the recovery path — it must be at least as guarded as promotion."""

    def _bundle(self, directory, schema, payload):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.pkl").write_bytes(payload)
        (directory / "metadata.json").write_text(
            json.dumps({"model_version": directory.name, "feature_schema": schema})
        )

    def _lineage(self, isolated_paths, versions):
        isolated_paths["lineage_path"].parent.mkdir(parents=True, exist_ok=True)
        isolated_paths["lineage_path"].write_text(
            json.dumps([{"model_version": v} for v in versions])
        )
        (isolated_paths["monitoring_dir"] / "health_report.json").write_text(
            json.dumps({"retraining_recommended": True})
        )

    def test_rollback_blocked_on_schema_mismatch(self, isolated_paths):
        """
        A rollback target with a different schema would break every request: the
        running API froze its request model from the schema now in production.
        """
        experiments = isolated_paths["experiments_dir"]
        production = isolated_paths["production_model"].parent

        self._bundle(experiments / "churn_model_20260101_000000", ["A", "B"], b"old")
        self._bundle(experiments / "churn_model_20260201_000000", ["A", "B", "C"], b"new")
        self._bundle(production, ["A", "B", "C"], b"new")
        self._lineage(
            isolated_paths,
            ["churn_model_20260101_000000", "churn_model_20260201_000000"],
        )

        assert rollback_if_needed() is False
        assert (production / "model.pkl").read_bytes() == b"new"

    def test_rollback_proceeds_when_schemas_match(self, isolated_paths):
        experiments = isolated_paths["experiments_dir"]
        production = isolated_paths["production_model"].parent

        self._bundle(experiments / "churn_model_20260101_000000", ["A", "B"], b"old")
        self._bundle(experiments / "churn_model_20260201_000000", ["A", "B"], b"new")
        self._bundle(production, ["A", "B"], b"new")
        self._lineage(
            isolated_paths,
            ["churn_model_20260101_000000", "churn_model_20260201_000000"],
        )

        assert rollback_if_needed() is True
        assert (production / "model.pkl").read_bytes() == b"old"


class TestRollbackNotifiesServing:
    """A rollback that no replica hears about has not recovered anything."""

    def test_orchestrator_reloads_serving_after_rollback(
        self, isolated_paths, monkeypatch
    ):
        import churn_system.lifecycle.orchestrator as orch

        calls = []
        monkeypatch.setattr(orch, "notify_serving_reload", lambda: calls.append(1))
        monkeypatch.setattr(orch, "evaluate_model_health", lambda: None)
        monkeypatch.setattr(orch, "rollback_if_needed", lambda: True)

        (isolated_paths["monitoring_dir"] / "health_report.json").write_text(
            json.dumps({"retraining_recommended": False})
        )

        outcome = orch.run_lifecycle()

        assert outcome["rolled_back"] is True
        assert calls == [1], "serving was not notified after rollback"
