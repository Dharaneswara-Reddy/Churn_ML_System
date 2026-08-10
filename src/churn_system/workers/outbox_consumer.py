"""
Distributed Outbox Consumer Worker.

Implements the Transactional Outbox Pattern for reliable event processing. This
worker runs as a SEPARATE PROCESS from the API server, claiming unprocessed rows
from the outbox table and publishing them.

Claiming model
--------------
Rows are claimed with a **lease**: a worker stamps ``claimed_at`` inside a single
committed transaction, and only rows that are unclaimed (or whose lease has
expired) are eligible. This gives at-least-once delivery with no duplicate
processing between healthy workers, and it works identically on SQLite and
PostgreSQL.

The previous implementation relied on ``SELECT ... FOR UPDATE SKIP LOCKED`` with a
try/except "SQLite fallback". That fallback was unreachable — the clause is built
lazily and SQLite's dialect simply drops it at compile time — so on SQLite two
workers claimed identical batches, and on PostgreSQL the session closed (releasing
the row locks) before processing even began.
"""

from __future__ import annotations

import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

from sqlalchemy import or_, select, update

from churn_system.config.config import CONFIG
from churn_system.events.db import OutboxEvent, SessionLocal, init_db, now_utc
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"].get("worker", "worker.log"))

# Configuration
POLL_INTERVAL = int(CONFIG.get("worker", {}).get("poll_interval_seconds", 5))
BATCH_SIZE = int(CONFIG.get("worker", {}).get("batch_size", 50))
MAX_WORKERS = int(CONFIG.get("worker", {}).get("max_workers", 4))
# How long a claim is honoured before another worker may retry the row.
CLAIM_LEASE_SECONDS = int(
    os.environ.get("CHURN_OUTBOX_LEASE_SECONDS", str(max(POLL_INTERVAL * 6, 60)))
)
MAX_ATTEMPTS = int(os.environ.get("CHURN_OUTBOX_MAX_ATTEMPTS", "5"))

# Graceful shutdown flag
_shutdown_event = threading.Event()


def _handle_signal(signum, frame):
    logger.info("Shutdown signal received (signal=%d) — draining current batch", signum)
    _shutdown_event.set()


def _process_single_event(event_id: int, event_type: str, payload: dict) -> bool:
    """
    Process a single outbox event.

    In production this would publish to a message broker (SNS/SQS/Kafka). For now it
    logs the event; returning False marks the event for retry.
    """
    logger.info(
        "Processing outbox event | id=%d | type=%s | request_id=%s",
        event_id,
        event_type,
        payload.get("request_id", "unknown"),
    )
    return True


def _claim_batch(limit: int) -> list[tuple[int, str, dict]]:
    """
    Atomically claim up to ``limit`` unprocessed events.

    The claim is a committed UPDATE, so a concurrent worker sees the stamped
    ``claimed_at`` and skips those rows. Rows whose lease has expired become
    claimable again, which is what makes a crashed worker's backlog recoverable.
    """
    cutoff = now_utc() - timedelta(seconds=CLAIM_LEASE_SECONDS)

    with SessionLocal() as session:
        candidates = session.execute(
            select(OutboxEvent.id)
            .where(
                OutboxEvent.processed_at.is_(None),
                OutboxEvent.attempts < MAX_ATTEMPTS,
                or_(
                    OutboxEvent.claimed_at.is_(None),
                    OutboxEvent.claimed_at < cutoff,
                ),
            )
            .order_by(OutboxEvent.created_at.asc())
            .limit(limit)
        ).scalars().all()

        if not candidates:
            return []

        claimed_at = now_utc()
        result = session.execute(
            update(OutboxEvent)
            .where(
                OutboxEvent.id.in_(candidates),
                OutboxEvent.processed_at.is_(None),
                or_(
                    OutboxEvent.claimed_at.is_(None),
                    OutboxEvent.claimed_at < cutoff,
                ),
            )
            .values(claimed_at=claimed_at, attempts=OutboxEvent.attempts + 1)
        )
        session.commit()

        if not result.rowcount:
            return []

        # Re-read only the rows this worker actually won.
        rows = session.execute(
            select(OutboxEvent).where(
                OutboxEvent.id.in_(candidates),
                OutboxEvent.claimed_at == claimed_at,
            )
        ).scalars().all()

        return [(row.id, row.event_type, dict(row.payload)) for row in rows]


def _mark_processed(event_ids: list[int]) -> None:
    if not event_ids:
        return
    with SessionLocal() as session:
        session.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id.in_(event_ids))
            .values(processed_at=now_utc())
        )
        session.commit()


def _release_claims(event_ids: list[int]) -> None:
    """Drop the lease on failed rows so they are retried promptly."""
    if not event_ids:
        return
    with SessionLocal() as session:
        session.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id.in_(event_ids))
            .values(claimed_at=None)
        )
        session.commit()


def _claim_and_process_batch() -> int:
    """
    Claim a batch of unprocessed outbox events, then process them concurrently.

    Returns the number of events successfully processed.
    """
    init_db()

    event_data = _claim_batch(BATCH_SIZE)
    if not event_data:
        return 0

    logger.info("Claimed %d outbox events for processing", len(event_data))

    processed_ids: list[int] = []
    failed_ids: list[int] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="outbox") as executor:
        future_to_id = {
            executor.submit(_process_single_event, eid, etype, payload): eid
            for eid, etype, payload in event_data
        }

        for future in as_completed(future_to_id):
            event_id = future_to_id[future]
            try:
                if future.result():
                    processed_ids.append(event_id)
                else:
                    failed_ids.append(event_id)
            except Exception:
                logger.exception("Failed to process outbox event id=%d", event_id)
                failed_ids.append(event_id)

    _mark_processed(processed_ids)
    if processed_ids:
        logger.info("Marked %d events as processed", len(processed_ids))

    _release_claims(failed_ids)
    if failed_ids:
        logger.warning(
            "Failed to process %d events (released for retry): %s",
            len(failed_ids),
            failed_ids,
        )

    return len(processed_ids)


def run_worker() -> None:
    """
    Main worker loop — polls the outbox table and processes events.

    Runs until SIGTERM/SIGINT. Multiple instances can run in parallel; the
    lease-based claim in ``_claim_batch`` keeps them from processing the same row.
    """
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        "Outbox consumer started | poll_interval=%ds | batch_size=%d | max_workers=%d "
        "| lease=%ds",
        POLL_INTERVAL,
        BATCH_SIZE,
        MAX_WORKERS,
        CLAIM_LEASE_SECONDS,
    )

    total_processed = 0

    while not _shutdown_event.is_set():
        try:
            count = _claim_and_process_batch()
            total_processed += count

            if count > 0:
                logger.info(
                    "Batch complete | processed=%d | total=%d", count, total_processed
                )
            else:
                logger.debug("No pending events — sleeping %ds", POLL_INTERVAL)

        except Exception:
            logger.exception("Worker batch processing failed — will retry")

        _shutdown_event.wait(timeout=POLL_INTERVAL)

    logger.info("Worker shutting down gracefully | total_processed=%d", total_processed)


if __name__ == "__main__":
    run_worker()
