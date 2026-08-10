"""
Outbox consumer tests.

The worker previously relied on ``SELECT ... FOR UPDATE SKIP LOCKED`` guarded by an
unreachable try/except. SQLite drops the locking clause at compile time, so two
workers claimed identical batches and published every event twice.
"""

from __future__ import annotations

import threading

import pytest

from churn_system.events.db import OutboxEvent, SessionLocal, init_db, now_utc
from churn_system.workers.outbox_consumer import (
    _claim_and_process_batch,
    _claim_batch,
)


@pytest.fixture
def pending_events():
    """Insert unprocessed outbox rows and return their ids."""
    init_db()
    with SessionLocal() as session:
        events = [
            OutboxEvent(
                created_at=now_utc(),
                event_type="prediction_made",
                payload={"request_id": f"req-{i}"},
                processed_at=None,
            )
            for i in range(10)
        ]
        session.add_all(events)
        session.commit()
        return [e.id for e in events]


class TestClaiming:
    def test_claim_returns_pending_events(self, pending_events):
        claimed = _claim_batch(limit=5)

        assert len(claimed) == 5
        assert all(isinstance(event_id, int) for event_id, _, _ in claimed)

    def test_second_claim_does_not_return_the_same_rows(self, pending_events):
        """A claimed row must be invisible to the next claim while its lease holds."""
        first = {event_id for event_id, _, _ in _claim_batch(limit=5)}
        second = {event_id for event_id, _, _ in _claim_batch(limit=5)}

        assert first, "first claim returned nothing"
        assert second, "second claim returned nothing"
        assert first.isdisjoint(second), f"rows claimed twice: {first & second}"

    def test_concurrent_workers_never_claim_the_same_row(self, pending_events):
        """
        Two workers racing on the same table must partition the work.

        This is the regression that matters: duplicate claims mean every downstream
        event is published more than once.
        """
        results: list[set[int]] = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait(timeout=5)
            claimed = {event_id for event_id, _, _ in _claim_batch(limit=10)}
            with lock:
                results.append(claimed)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 2
        overlap = results[0] & results[1]
        assert not overlap, f"both workers claimed {overlap}"
        assert len(results[0] | results[1]) == len(pending_events)

    def test_exhausted_events_are_not_reclaimed(self, pending_events, monkeypatch):
        """A permanently failing event must stop after MAX_ATTEMPTS."""
        import churn_system.workers.outbox_consumer as worker_mod

        monkeypatch.setattr(worker_mod, "MAX_ATTEMPTS", 1)
        monkeypatch.setattr(worker_mod, "CLAIM_LEASE_SECONDS", 0)

        first = _claim_batch(limit=10)
        assert first, "expected an initial claim"

        assert _claim_batch(limit=10) == []


class TestProcessing:
    def test_processed_events_are_marked_and_not_reprocessed(self, pending_events):
        processed = _claim_and_process_batch()

        assert processed == len(pending_events)
        assert _claim_and_process_batch() == 0

    def test_failed_events_are_released_for_retry(self, pending_events, monkeypatch):
        import churn_system.workers.outbox_consumer as worker_mod

        monkeypatch.setattr(worker_mod, "_process_single_event", lambda *a: False)

        assert worker_mod._claim_and_process_batch() == 0

        # The lease was dropped, so the rows are immediately claimable again.
        assert _claim_batch(limit=10)

    def test_empty_outbox_is_a_noop(self):
        assert _claim_and_process_batch() == 0
