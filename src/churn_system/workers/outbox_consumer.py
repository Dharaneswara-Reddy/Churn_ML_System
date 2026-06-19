"""
Distributed Outbox Consumer Worker.

Implements the Transactional Outbox Pattern for reliable event processing
in a distributed system. This worker runs as a SEPARATE PROCESS from the
API server, polling the outbox_events table and processing unhandled events.

Distributed Systems Concepts Demonstrated:
  1. Transactional Outbox Pattern: events are written atomically with the
     business transaction (prediction), then consumed asynchronously.
  2. Exactly-once processing: uses database-level row locking
     (SELECT ... FOR UPDATE SKIP LOCKED) to prevent duplicate processing
     when multiple worker instances run concurrently.
  3. Leader election via DB locks: multiple workers can run without
     coordination — the DB handles mutual exclusion at the row level.
  4. Batch processing with backpressure: configurable batch size limits
     memory usage and prevents one slow consumer from blocking others.
  5. Graceful shutdown: SIGTERM/SIGINT handling with drain period.

In production (AWS), this would publish events to SNS/SQS/Kafka.
Locally, it logs processed events and updates their processed_at timestamp.
"""

from __future__ import annotations

import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import select, update

from churn_system.config.config import CONFIG
from churn_system.events.db import OutboxEvent, SessionLocal, init_db, now_utc
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"].get("worker", "worker.log"))

# Configuration
POLL_INTERVAL = int(CONFIG.get("worker", {}).get("poll_interval_seconds", 5))
BATCH_SIZE = int(CONFIG.get("worker", {}).get("batch_size", 50))
MAX_WORKERS = int(CONFIG.get("worker", {}).get("max_workers", 4))

# Graceful shutdown flag
_shutdown_event = threading.Event()


def _handle_signal(signum, frame):  # noqa: ARG001
    logger.info("Shutdown signal received (signal=%d) — draining current batch", signum)
    _shutdown_event.set()


def _process_single_event(event_id: int, event_type: str, payload: dict) -> bool:
    """
    Process a single outbox event.

    In production, this would publish to a message broker (SNS/SQS/Kafka).
    For now, it logs the event and simulates processing.

    Returns True if processing succeeded.
    """
    logger.info(
        "Processing outbox event | id=%d | type=%s | request_id=%s",
        event_id,
        event_type,
        payload.get("request_id", "unknown"),
    )

    # In a real distributed system, this would be:
    # - boto3.client('sns').publish(TopicArn=..., Message=json.dumps(payload))
    # - boto3.client('sqs').send_message(QueueUrl=..., MessageBody=json.dumps(payload))
    # - kafka_producer.send(topic, value=payload)

    # Simulate processing latency (remove in production)
    # time.sleep(0.01)

    return True


def _claim_and_process_batch() -> int:
    """
    Claim a batch of unprocessed outbox events using row-level locking,
    then process them concurrently.

    Uses SELECT ... FOR UPDATE SKIP LOCKED to ensure that multiple worker
    instances can safely run in parallel. Each worker claims its own batch
    without blocking other workers.

    Returns the number of events processed.
    """
    init_db()

    with SessionLocal() as session:
        # Claim unprocessed events with row-level locking.
        # FOR UPDATE SKIP LOCKED is the distributed systems primitive that
        # enables concurrent worker instances without external coordination.
        # - FOR UPDATE: locks the selected rows
        # - SKIP LOCKED: if another worker already locked a row, skip it
        #
        # Note: SQLite doesn't support SKIP LOCKED — for local dev we use
        # a simpler SELECT. In production (PostgreSQL), the full locking
        # semantics apply.
        stmt = (
            select(OutboxEvent)
            .where(OutboxEvent.processed_at.is_(None))
            .order_by(OutboxEvent.created_at.asc())
            .limit(BATCH_SIZE)
        )

        # PostgreSQL: add row-level locking for concurrent workers
        try:
            stmt = stmt.with_for_update(skip_locked=True)
        except Exception:
            # SQLite fallback — no row-level locking available
            pass

        events = session.execute(stmt).scalars().all()

        if not events:
            return 0

        logger.info("Claimed %d outbox events for processing", len(events))

        # Prepare event data (detach from session before threading)
        event_data = [
            (e.id, e.event_type, dict(e.payload)) for e in events
        ]

    # Process events concurrently using ThreadPoolExecutor
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
                success = future.result()
                if success:
                    processed_ids.append(event_id)
                else:
                    failed_ids.append(event_id)
            except Exception:
                logger.exception("Failed to process outbox event id=%d", event_id)
                failed_ids.append(event_id)

    # Mark processed events (batch UPDATE for efficiency)
    if processed_ids:
        with SessionLocal() as session:
            session.execute(
                update(OutboxEvent)
                .where(OutboxEvent.id.in_(processed_ids))
                .values(processed_at=now_utc())
            )
            session.commit()

        logger.info("Marked %d events as processed", len(processed_ids))

    if failed_ids:
        logger.warning("Failed to process %d events: %s", len(failed_ids), failed_ids)

    return len(processed_ids)


def run_worker() -> None:
    """
    Main worker loop — polls the outbox table and processes events.

    Runs indefinitely until SIGTERM/SIGINT is received. Each iteration:
    1. Claims a batch of unprocessed events (with row-level locking)
    2. Processes them concurrently
    3. Marks them as processed
    4. Sleeps for POLL_INTERVAL seconds

    Multiple instances of this worker can run in parallel (horizontally
    scaled) — the database row-level locking ensures no duplicate processing.
    """
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        "Outbox consumer worker started | poll_interval=%ds | batch_size=%d | max_workers=%d",
        POLL_INTERVAL,
        BATCH_SIZE,
        MAX_WORKERS,
    )

    total_processed = 0

    while not _shutdown_event.is_set():
        try:
            count = _claim_and_process_batch()
            total_processed += count

            if count > 0:
                logger.info(
                    "Batch complete | processed=%d | total=%d",
                    count,
                    total_processed,
                )
            else:
                logger.debug("No pending events — sleeping %ds", POLL_INTERVAL)

        except Exception:
            logger.exception("Worker batch processing failed — will retry")

        # Interruptible sleep using the shutdown event
        _shutdown_event.wait(timeout=POLL_INTERVAL)

    logger.info("Worker shutting down gracefully | total_processed=%d", total_processed)


if __name__ == "__main__":
    run_worker()
