"""Unit tests for lineage tracking."""

from __future__ import annotations

import json


def test_record_lineage_appends_record(tmp_path, monkeypatch):
    """record_lineage should append to the lineage file."""
    from churn_system.config import config as cfg

    lineage_path = tmp_path / "lineage.json"
    monkeypatch.setitem(cfg.CONFIG["paths"], "lineage_path", str(lineage_path))

    # Re-import to pick up the patched path
    import churn_system.lifecycle.lineage as lineage_mod
    monkeypatch.setattr(lineage_mod, "LINEAGE_PATH", lineage_path)
    lineage_path.parent.mkdir(parents=True, exist_ok=True)

    lineage_mod.record_lineage(
        model_version="v1",
        metrics={"roc_auc": 0.85},
        dataset_used="data/test.csv",
        trigger="manual",
        parent_model=None,
    )

    data = json.loads(lineage_path.read_text())
    assert len(data) == 1
    assert data[0]["model_version"] == "v1"
    assert data[0]["trigger"] == "manual"

    # Append a second record
    lineage_mod.record_lineage(
        model_version="v2",
        metrics={"roc_auc": 0.87},
        dataset_used="data/test2.csv",
        trigger="drift_retraining",
        parent_model="v1",
    )

    data = json.loads(lineage_path.read_text())
    assert len(data) == 2
    assert data[1]["parents_model"] == "v1"


def test_load_lineage_returns_empty_for_missing_file(tmp_path, monkeypatch):
    """load_lineage should return [] if the file does not exist."""
    import churn_system.lifecycle.lineage as lineage_mod

    missing_path = tmp_path / "nonexistent.json"
    monkeypatch.setattr(lineage_mod, "LINEAGE_PATH", missing_path)

    assert lineage_mod.load_lineage() == []
