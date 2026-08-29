"""Regression: training orchestrator runs end-to-end on synthetic data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_training_pipeline_main_smoke(tmp_path, monkeypatch):
    from churn_system.config import config as cfg

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "generate_smoke_csv.py"
    assert script.exists(), "scripts/generate_smoke_csv.py missing"

    csv_path = tmp_path / "smoke.csv"
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(script), str(csv_path), "--rows", "300"],
        check=True,
        cwd=str(root),
    )

    exp_dir = tmp_path / "experiments"
    ref_path = tmp_path / "training_reference.csv"

    monkeypatch.setitem(cfg.CONFIG["paths"], "raw_data", str(csv_path))
    monkeypatch.setitem(cfg.CONFIG["paths"], "experiments_dir", str(exp_dir))
    monkeypatch.setitem(cfg.CONFIG["paths"], "training_reference", str(ref_path))
    monkeypatch.setenv("CHURN_MLFLOW_ENABLED", "0")

    from churn_system.training.train import main

    main()

    versions = sorted(exp_dir.glob("churn_model_*"))
    assert versions, "No experiment directory written"
    latest = versions[-1]
    assert (latest / "model.pkl").exists()
    assert (latest / "metadata.json").exists()
    meta = json.loads((latest / "metadata.json").read_text())
    assert "feature_types" in meta
    assert meta.get("model_type")


class TestDatasetSelectionPrecedence:
    """
    Which CSV training actually reads.

    An explicit ``CHURN_RAW_DATA_PATH`` used to lose to the retraining-dataset
    preference. The consequence was not subtle: the CI smoke job, configured to
    train on a synthetic CSV precisely so it would never touch customer data,
    trained on ``data/retraining_dataset.csv`` on any machine where that file
    existed. The override was accepted, logged, and then ignored.
    """

    def test_an_explicit_override_beats_the_retraining_dataset(
        self, tmp_path, monkeypatch
    ):
        from churn_system.config.config import CONFIG
        from churn_system.training.steps.data_ingestion import resolve_training_data_path

        retraining = tmp_path / "retraining.csv"
        retraining.write_text("a,b\n1,2\n")
        explicit = tmp_path / "explicit.csv"
        explicit.write_text("a,b\n3,4\n")

        monkeypatch.setitem(CONFIG["paths"], "retraining_data", str(retraining))
        monkeypatch.setitem(CONFIG["paths"], "raw_data", str(explicit))
        monkeypatch.setenv("CHURN_RAW_DATA_PATH", str(explicit))

        assert resolve_training_data_path() == explicit

    def test_without_an_override_the_retraining_dataset_still_wins(
        self, tmp_path, monkeypatch
    ):
        """
        The lifecycle depends on this: without the preference, every drift cycle
        wrote a retraining dataset that nothing ever read, so "retraining on fresh
        data" silently re-fit the original static CSV.
        """
        from churn_system.config.config import CONFIG
        from churn_system.training.steps.data_ingestion import resolve_training_data_path

        retraining = tmp_path / "retraining.csv"
        retraining.write_text("a,b\n1,2\n")
        raw = tmp_path / "raw.csv"
        raw.write_text("a,b\n3,4\n")

        monkeypatch.setitem(CONFIG["paths"], "retraining_data", str(retraining))
        monkeypatch.setitem(CONFIG["paths"], "raw_data", str(raw))
        monkeypatch.delenv("CHURN_RAW_DATA_PATH", raising=False)

        assert resolve_training_data_path() == retraining

    def test_the_raw_dataset_is_the_final_fallback(self, tmp_path, monkeypatch):
        from churn_system.config.config import CONFIG
        from churn_system.training.steps.data_ingestion import resolve_training_data_path

        raw = tmp_path / "raw.csv"
        raw.write_text("a,b\n3,4\n")

        monkeypatch.setitem(
            CONFIG["paths"], "retraining_data", str(tmp_path / "absent.csv")
        )
        monkeypatch.setitem(CONFIG["paths"], "raw_data", str(raw))
        monkeypatch.delenv("CHURN_RAW_DATA_PATH", raising=False)

        assert resolve_training_data_path() == raw

    def test_an_empty_override_does_not_count_as_explicit(self, tmp_path, monkeypatch):
        """
        Compose passes unset variables through as empty strings, so an empty value
        must not be read as "the operator chose this path".
        """
        from churn_system.config.config import CONFIG
        from churn_system.training.steps.data_ingestion import resolve_training_data_path

        retraining = tmp_path / "retraining.csv"
        retraining.write_text("a,b\n1,2\n")

        monkeypatch.setitem(CONFIG["paths"], "retraining_data", str(retraining))
        monkeypatch.setitem(CONFIG["paths"], "raw_data", str(tmp_path / "raw.csv"))
        monkeypatch.setenv("CHURN_RAW_DATA_PATH", "")

        assert resolve_training_data_path() == retraining


class TestTrainingSignsItsOutput:
    """
    Training signs the bundle it writes.

    Signing only at promotion left a gap in the chain of custody: promotion blessed
    whatever it found in ``models/experiments/`` without verifying it, so anyone
    able to write there could tamper with a ``model.pkl`` and have promotion sign
    it — after which the API would verify the signature happily and unpickle it.
    """

    def test_training_writes_a_valid_signature(self, tmp_path, monkeypatch):
        import subprocess
        import sys

        from churn_system.artifacts import verify_bundle_signature

        raw = tmp_path / "smoke.csv"
        subprocess.run(
            [sys.executable, "scripts/generate_smoke_csv.py", str(raw), "--rows", "300"],
            check=True,
            capture_output=True,
        )

        experiments = tmp_path / "experiments"
        env = {
            **os.environ,
            "PYTHONPATH": "src",
            "CHURN_RAW_DATA_PATH": str(raw),
            "CHURN_EXPERIMENTS_DIR": str(experiments),
            "CHURN_TRAINING_REFERENCE_PATH": str(tmp_path / "reference.csv"),
            "CHURN_MLFLOW_ENABLED": "0",
            "CHURN_ARTIFACT_SIGNING_KEY": "training-signs-its-output",
        }
        subprocess.run(
            [sys.executable, "-m", "churn_system.training.train"],
            check=True,
            capture_output=True,
            env=env,
        )

        bundles = list(experiments.glob("churn_model_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "signature.json").exists()

        monkeypatch.setenv("CHURN_ARTIFACT_SIGNING_KEY", "training-signs-its-output")
        verify_bundle_signature(bundles[0])  # must not raise

    def test_promotion_refuses_an_unsigned_source(self, tmp_path, monkeypatch):
        import json

        from churn_system.artifacts import ArtifactSignatureError, swap_model_bundle

        monkeypatch.setenv("CHURN_ARTIFACT_SIGNING_KEY", "a-real-key")
        monkeypatch.delenv("CHURN_ALLOW_UNSIGNED_ARTIFACTS", raising=False)

        source = tmp_path / "tampered"
        source.mkdir()
        (source / "model.pkl").write_bytes(b"not really a model")
        (source / "metadata.json").write_text(json.dumps({"model_version": "evil"}))

        with pytest.raises(ArtifactSignatureError):
            swap_model_bundle(source, tmp_path / "production", sign=True)

        assert not (tmp_path / "production").exists(), (
            "An unsigned bundle reached the production slot."
        )
