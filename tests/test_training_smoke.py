"""Regression: training orchestrator runs end-to-end on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path


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
