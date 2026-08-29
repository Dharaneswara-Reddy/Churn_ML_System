"""
Event store retention.

Both tables grew without bound. Measured on this schema: 200,000 prediction rows
took 15.5s and 766MB peak to load, extrapolating to roughly 38GB at 10M rows —
and the automated health check loads them on every lifecycle cycle. That makes
unbounded growth a slow-motion outage of the self-healing system itself.

Design constraints, in priority order:

1. **Never delete work that has not been done.** PENDING and PROCESSING rows are
   live; DEAD_LETTER rows are evidence of a failure someone still needs to see.
2. **Keep dead letters longer than successes.** A processed event is disposable;
   a permanently failed one is a defect report.
3. **Bounded batches.** A single unbounded DELETE against millions of rows holds
   a long write transaction, which on SQLite blocks every writer in the process
   and on PostgreSQL bloats the table. Each batch is committed separately.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import delete, func, select

from churn_system.config.config import CONFIG
from churn_system.events.db import (
    OutboxEvent,
    OutboxStatus,
    PredictionEvent,
    SessionLocal,
    init_db,
    now_utc,
)
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"].get("worker", "worker.log"))

DEFAULT_PROCESSED_RETENTION_DAYS = 7
DEFAULT_DEAD_LETTER_RETENTION_DAYS = 90
DEFAULT_PREDICTION_RETENTION_DAYS = 180
DEFAULT_BATCH_SIZE = 1000


def _retention_config() -> dict[str, Any]:
    return dict(CONFIG.get("retention", {}))


def _setting(key: str, default: int) -> int:
    return int(_retention_config().get(key, default))


def _delete_in_batches(model, condition, batch_size: int, label: str) -> int:
    """
    Delete rows matching ``condition`` in committed batches.

    Selects a bounded set of primary keys, deletes exactly those, commits, and
    repeats — so no single transaction is held open across the whole table.
    """
    total = 0
    while True:
        with SessionLocal() as session:
            ids = session.execute(
                select(model.id).where(condition).limit(batch_size)
            ).scalars().all()

            if not ids:
                break

            session.execute(delete(model).where(model.id.in_(ids)))
            session.commit()
            total += len(ids)

        if len(ids) < batch_size:
            break

    if total:
        logger.info("Retention: deleted %d %s rows", total, label)
    return total


def purge_processed_outbox_events(batch_size: int | None = None) -> int:
    """Delete PROCESSED outbox events older than the configured retention."""
    init_db()
    days = _setting("processed_outbox_days", DEFAULT_PROCESSED_RETENTION_DAYS)
    cutoff = now_utc() - timedelta(days=days)

    return _delete_in_batches(
        OutboxEvent,
        (OutboxEvent.status == OutboxStatus.PROCESSED.value)
        & (OutboxEvent.processed_at.is_not(None))
        & (OutboxEvent.processed_at < cutoff),
        batch_size or _setting("batch_size", DEFAULT_BATCH_SIZE),
        "processed outbox",
    )


def purge_dead_letter_events(batch_size: int | None = None) -> int:
    """
    Delete DEAD_LETTER events older than their (longer) retention.

    Kept far longer than successes because they are the only durable record that
    an event could never be delivered.
    """
    init_db()
    days = _setting("dead_letter_days", DEFAULT_DEAD_LETTER_RETENTION_DAYS)
    cutoff = now_utc() - timedelta(days=days)

    return _delete_in_batches(
        OutboxEvent,
        (OutboxEvent.status == OutboxStatus.DEAD_LETTER.value)
        & (OutboxEvent.created_at < cutoff),
        batch_size or _setting("batch_size", DEFAULT_BATCH_SIZE),
        "dead-letter outbox",
    )


def purge_old_predictions(batch_size: int | None = None) -> int:
    """
    Delete unlabelled prediction events past the retention window.

    Labelled rows are deliberately retained regardless of age: they are training
    data, and deleting them would silently shrink the retraining set.
    """
    init_db()
    days = _setting("prediction_days", DEFAULT_PREDICTION_RETENTION_DAYS)
    cutoff = now_utc() - timedelta(days=days)

    return _delete_in_batches(
        PredictionEvent,
        (PredictionEvent.created_at < cutoff) & (PredictionEvent.label.is_(None)),
        batch_size or _setting("batch_size", DEFAULT_BATCH_SIZE),
        "old prediction",
    )


def outbox_backlog() -> dict[str, int]:
    """
    Count outbox events by status — the numbers an operator actually needs.

    ``dead_letter`` in particular was previously invisible: those rows looked
    identical to pending work in any naive query.
    """
    init_db()
    with SessionLocal() as session:
        rows = session.execute(
            select(OutboxEvent.status, func.count(OutboxEvent.id)).group_by(
                OutboxEvent.status
            )
        ).all()

    counts = {status.value: 0 for status in OutboxStatus}
    for status, count in rows:
        counts[str(status)] = int(count)
    return counts


def run_retention() -> dict[str, int]:
    """Run every retention task. Safe to call repeatedly; returns rows deleted."""
    return {
        "processed_outbox": purge_processed_outbox_events(),
        "dead_letter_outbox": purge_dead_letter_events(),
        "old_predictions": purge_old_predictions(),
    }
