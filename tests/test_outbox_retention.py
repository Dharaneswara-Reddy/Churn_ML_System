"""
Outbox status, dead-letter, and retention tests.

Two defects these pin down:

* An event that exhausted its retry budget stayed ``processed_at IS NULL`` with a
  full attempt count — indistinguishable from live pending work, so it was
  invisible to every status query and metric.
* Neither table had retention. 200k prediction rows measured at 15.5s and 766MB
  peak to load, and the automated health check loads them every cycle, so
  unbounded growth was a slow-motion outage of the self-healing loop.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from churn_system.events.db import (
    OutboxEvent,
    OutboxStatus,
    PredictionEvent,
    SessionLocal,
    init_db,
    now_utc,
)
from churn_system.events.retention import (
    outbox_backlog,
    purge_dead_letter_events,
    purge_old_predictions,
    purge_processed_outbox_events,
)


def _add_outbox(status, *, age_days=0, processed=False, attempts=0):
    init_db()
    created = now_utc() - timedelta(days=age_days)
    with SessionLocal() as session:
        event = OutboxEvent(
            created_at=created,
            event_type="prediction_made",
            payload={"request_id": "r"},
            processed_at=created if processed else None,
            attempts=attempts,
            status=status.value,
        )
        session.add(event)
        session.commit()
        return event.id


def _add_prediction(*, age_days=0, label=None):
    init_db()
    with SessionLocal() as session:
        event = PredictionEvent(
            request_id=f"req-{age_days}-{label}",
            created_at=now_utc() - timedelta(days=age_days),
            model_version="v1",
            probability=0.5,
            prediction=1,
            latency_seconds=0.01,
            features={},
            label=label,
        )
        session.add(event)
        session.commit()
        return event.id


class TestDeadLetterIsExplicit:
    def test_backlog_distinguishes_dead_letter_from_pending(self):
        """
        The whole point: a permanently failed event must not be counted as
        outstanding work an operator is waiting on.
        """
        _add_outbox(OutboxStatus.PENDING)
        _add_outbox(OutboxStatus.DEAD_LETTER, attempts=5)

        backlog = outbox_backlog()

        assert backlog["PENDING"] == 1
        assert backlog["DEAD_LETTER"] == 1

    def test_exhausted_event_is_dead_lettered_by_the_worker(self, monkeypatch):
        import churn_system.workers.outbox_consumer as worker

        monkeypatch.setattr(worker, "MAX_ATTEMPTS", 1)
        monkeypatch.setattr(worker, "_process_single_event", lambda *a: False)
        _add_outbox(OutboxStatus.PENDING)

        worker._claim_and_process_batch()

        assert outbox_backlog()["DEAD_LETTER"] == 1

    def test_dead_lettered_event_records_a_reason(self, monkeypatch):
        import churn_system.workers.outbox_consumer as worker

        monkeypatch.setattr(worker, "MAX_ATTEMPTS", 1)
        monkeypatch.setattr(worker, "_process_single_event", lambda *a: False)
        _add_outbox(OutboxStatus.PENDING)

        worker._claim_and_process_batch()

        with SessionLocal() as session:
            event = session.query(OutboxEvent).first()
            assert event.status == OutboxStatus.DEAD_LETTER.value
            assert event.last_error


class TestRetentionNeverDeletesLiveWork:
    def test_pending_events_are_never_deleted(self):
        _add_outbox(OutboxStatus.PENDING, age_days=9999)

        purge_processed_outbox_events()
        purge_dead_letter_events()

        assert outbox_backlog()["PENDING"] == 1

    def test_processing_events_are_never_deleted(self):
        _add_outbox(OutboxStatus.PROCESSING, age_days=9999)

        purge_processed_outbox_events()

        assert outbox_backlog()["PROCESSING"] == 1

    def test_dead_letters_survive_the_processed_retention_window(self):
        """Dead letters have their own, longer retention — a failure record."""
        _add_outbox(OutboxStatus.DEAD_LETTER, age_days=30, attempts=5)

        purge_processed_outbox_events()

        assert outbox_backlog()["DEAD_LETTER"] == 1


class TestRetentionDeletesWhatItShould:
    def test_old_processed_events_are_deleted(self):
        _add_outbox(OutboxStatus.PROCESSED, age_days=90, processed=True)

        deleted = purge_processed_outbox_events()

        assert deleted == 1
        assert outbox_backlog()["PROCESSED"] == 0

    def test_recent_processed_events_are_kept(self):
        _add_outbox(OutboxStatus.PROCESSED, age_days=0, processed=True)

        purge_processed_outbox_events()

        assert outbox_backlog()["PROCESSED"] == 1

    def test_very_old_dead_letters_are_eventually_deleted(self):
        _add_outbox(OutboxStatus.DEAD_LETTER, age_days=500, attempts=5)

        assert purge_dead_letter_events() == 1

    def test_deletion_is_batched(self, monkeypatch):
        """
        A single unbounded DELETE holds a long write transaction, which on SQLite
        blocks every other writer in the process.
        """
        for _ in range(5):
            _add_outbox(OutboxStatus.PROCESSED, age_days=90, processed=True)

        deleted = purge_processed_outbox_events(batch_size=2)

        assert deleted == 5  # all removed, but across several committed batches


class TestPredictionRetention:
    def test_labelled_predictions_are_never_deleted(self):
        """Labelled rows are training data; ageing them out shrinks the dataset."""
        _add_prediction(age_days=9999, label=1)

        purge_old_predictions()

        with SessionLocal() as session:
            assert session.query(PredictionEvent).count() == 1

    def test_old_unlabelled_predictions_are_deleted(self):
        _add_prediction(age_days=9999, label=None)

        assert purge_old_predictions() == 1

    def test_recent_unlabelled_predictions_are_kept(self):
        _add_prediction(age_days=1, label=None)

        purge_old_predictions()

        with SessionLocal() as session:
            assert session.query(PredictionEvent).count() == 1


class TestBoundedMonitoringRead:
    def test_reader_is_bounded_by_default(self, monkeypatch):
        """An unbounded read is what made the health check OOM as the table grew."""
        from churn_system.config import config as cfg
        from churn_system.monitoring import prediction_reader

        monkeypatch.setitem(cfg.CONFIG["monitoring"], "max_production_rows", 3)
        for i in range(6):
            _add_prediction(age_days=i, label=None)

        frame = prediction_reader.load_predictions_df()

        assert len(frame) == 3

    def test_explicit_zero_requests_everything(self, monkeypatch):
        from churn_system.monitoring import prediction_reader

        for i in range(4):
            _add_prediction(age_days=i, label=None)

        assert len(prediction_reader.load_predictions_df(limit=0)) == 4


@pytest.fixture(autouse=True)
def _clean_tables():
    """Retention tests assert on absolute counts, so start from empty."""
    init_db()
    from sqlalchemy import delete

    from churn_system.events.db import ENGINE

    with ENGINE.begin() as conn:
        conn.execute(delete(OutboxEvent))
        conn.execute(delete(PredictionEvent))
    yield
