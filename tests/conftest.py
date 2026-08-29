"""
Shared test configuration and isolation fixtures.

Two kinds of setup live here:

1. **Environment, applied at import time.** Several modules read configuration and
   build objects (the event-store engine, logger paths, module-level constants) at
   *import*, and pytest imports test modules during collection — before any fixture
   runs. Setting these in a fixture would therefore be too late, so they are set
   when this file is imported, which happens first.

2. **Per-test isolation, applied as autouse fixtures.** The model registry, the
   model-contract cache and the Prometheus registry are process-global; without an
   explicit reset, state leaks between tests and results depend on execution order.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# --- import-time environment -------------------------------------------------
# A dedicated event store per test session: without this the suite writes real rows
# into data/churn_events.db, which both pollutes production data and makes
# `events/db.py` look covered when nothing asserts on it.
_TEST_STATE_DIR = Path(tempfile.mkdtemp(prefix="churn-tests-"))

os.environ.setdefault(
    "CHURN_EVENT_STORE_DATABASE_URL",
    f"sqlite:///{_TEST_STATE_DIR / 'events.db'}",
)
# The API refuses to start unauthenticated unless this is an explicit choice.
os.environ.setdefault("CHURN_ALLOW_ANONYMOUS", "1")
os.environ.setdefault("CHURN_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("CHURN_MLFLOW_ENABLED", "0")
# Model bundles are HMAC-signed and verification fails closed, so the suite needs a
# key. A fixed test key is correct here: these tests must exercise the real
# signing/verification path, not an opt-out that would let an unsigned bundle pass.
os.environ.setdefault("CHURN_ARTIFACT_SIGNING_KEY", "test-artifact-signing-key")
# Pseudonymisation also fails closed without a salt.
os.environ.setdefault("CHURN_SUBJECT_KEY_SALT", "test-subject-key-salt")


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Reset process-global singletons and caches around every test."""
    from churn_system.inference.model_contract import clear_model_contract_cache
    from churn_system.serving.model_registry import ModelRegistry

    ModelRegistry.reset()
    clear_model_contract_cache()

    yield

    ModelRegistry.reset()
    clear_model_contract_cache()


@pytest.fixture(autouse=True)
def _isolate_event_store():
    """
    Empty the prediction/outbox tables between tests.

    The engine is built once at import from a single URL, so every test shares one
    database. Without truncation, rows written by the API tests are still there when
    a monitoring test asks the event store for "production data", which makes those
    tests pass or fail depending on execution order.
    """
    yield

    from sqlalchemy import delete

    from churn_system.events.db import ENGINE, OutboxEvent, PredictionEvent, init_db

    init_db()
    with ENGINE.begin() as conn:
        conn.execute(delete(PredictionEvent))
        conn.execute(delete(OutboxEvent))


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """
    Redirect every configured filesystem path into ``tmp_path``.

    Tests that exercise training, promotion, rollback or monitoring touch real
    directories otherwise — including ``models/production/current``, which the
    lifecycle code deletes and recreates.
    """
    from churn_system.config import config as cfg

    layout = {
        "raw_data": tmp_path / "raw.csv",
        "retraining_data": tmp_path / "retraining.csv",
        "training_reference": tmp_path / "training_reference.csv",
        "production_model": tmp_path / "production" / "current" / "model.pkl",
        "experiments_dir": tmp_path / "experiments",
        "monitoring_dir": tmp_path / "monitoring",
        "lineage_path": tmp_path / "lineage" / "lineage.json",
        "prediction_log_csv": tmp_path / "inference_logs" / "predictions.csv",
    }

    for key, value in layout.items():
        monkeypatch.setitem(cfg.CONFIG["paths"], key, str(value))

    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "monitoring").mkdir(parents=True, exist_ok=True)

    return layout


@pytest.fixture
def propagating_logger():
    """
    Let ``caplog`` observe project loggers.

    ``get_logger`` sets ``propagate = False`` so subsystem logs stay in their own
    files, which also means pytest's caplog handler never sees them.
    """
    import logging

    toggled: list[logging.Logger] = []

    def _enable(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.propagate = True
        toggled.append(logger)
        return logger

    yield _enable

    for logger in toggled:
        logger.propagate = False
