"""
Durable event storage (SQLite by default).

This replaces CSV append logging for inference events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from churn_system.config.config import load_config


def _db_url() -> str:
    cfg = load_config()
    return str(cfg.get("event_store", {}).get("database_url", "sqlite:///./data/churn_events.db"))


def _engine_options(url: str) -> dict[str, Any]:
    """
    Connection options appropriate to the backend.

    SQLite is a local file with one writer; pooling options are meaningless and
    some are rejected outright. Managed PostgreSQL drops idle connections
    (RDS idle timeouts, PgBouncer recycling), so without ``pool_pre_ping`` the
    first query after an idle period fails with a stale-connection
    ``OperationalError`` instead of transparently reconnecting.
    """
    if url.startswith("sqlite"):
        return {}

    return {
        # Validate a pooled connection before handing it out.
        "pool_pre_ping": True,
        # Recycle below typical proxy/server idle timeouts.
        "pool_recycle": 1800,
        # Sized for the API's event-writer pool (4) plus monitoring reads, with
        # headroom for a second replica sharing the same database.
        "pool_size": 10,
        "max_overflow": 5,
    }


def _create_engine():
    """
    Build the engine, failing with a message that names the cause.

    SQLAlchemy imports the DBAPI eagerly, so a missing driver surfaces as a bare
    ``ModuleNotFoundError`` naming ``psycopg`` — with no hint that the trigger was
    ``CHURN_EVENT_STORE_DATABASE_URL``, and it fires at *import* of this module,
    which the API imports at module scope. That turned the single documented
    scaling step into an immediate crash of the API, worker and monitoring reader.
    """
    url = _db_url()
    try:
        return create_engine(url, future=True, **_engine_options(url))
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Database driver for CHURN_EVENT_STORE_DATABASE_URL={url!r} is not "
            f"installed ({exc}). For PostgreSQL install the extra: "
            'pip install "churn-ml-system[postgres]"'
        ) from exc


ENGINE = _create_engine()
SessionLocal = sessionmaker(bind=ENGINE, autocommit=False, autoflush=False, future=True)

# JSONB on PostgreSQL (indexable, binary, typed) while remaining plain JSON on
# SQLite, which has no JSONB. Declared as a variant so one column definition is
# correct on both backends.
JSONType = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    metadata = MetaData()


class OutboxStatus(str, Enum):
    """
    Explicit lifecycle state for an outbox event.

    Previously an exhausted event sat at ``attempts == MAX_ATTEMPTS`` with
    ``processed_at IS NULL`` — indistinguishable from a normal pending row, so a
    "pending count" metric silently included permanently-stuck events and nothing
    surfaced them. Making the state explicit means a dead-lettered event can be
    counted, alerted on, and retained separately from live work.
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    DEAD_LETTER = "DEAD_LETTER"


class PredictionEvent(Base):
    __tablename__ = "prediction_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    probability: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)
    latency_seconds: Mapped[float] = mapped_column(Float)

    # Redacted features only (no CustomerID / geo)
    features: Mapped[dict[str, Any]] = mapped_column(JSONType)

    # Ground truth, supplied later via the feedback endpoint. Without this the
    # system can only measure input drift and can never tell whether a past
    # prediction was actually correct.
    label: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    labeled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Pseudonymous, salted subject key. Lets a data-subject erasure request find
    # every row for a customer without storing the customer id itself.
    subject_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class OutboxEvent(Base):
    """
    Simple queue-like outbox table for async processors (Kafka/SQS later).
    """

    __tablename__ = "outbox_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Lease-based claiming. A worker stamps claimed_at to take ownership of a row;
    # the claim expires after a lease interval so a crashed worker's rows are
    # retried rather than stranded. Without this, "claiming" relied on
    # SELECT ... FOR UPDATE SKIP LOCKED, which SQLite silently ignores — two
    # workers would then process every event twice.
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # Explicit state, indexed because both the claim query and the retention job
    # filter on it. `processed_at` alone could not distinguish "waiting" from
    # "permanently failed".
    status: Mapped[str] = mapped_column(
        String(16),
        default=OutboxStatus.PENDING.value,
        server_default=OutboxStatus.PENDING.value,
        index=True,
    )
    # Why the event was dead-lettered, so an operator does not have to reconstruct
    # it from logs that may have rotated away.
    last_error: Mapped[str | None] = mapped_column(String(512), nullable=True)


_SCHEMA_READY = False


# The claim query filters on (status, attempts, claimed_at) and orders by
# created_at; the retention job filters on (status, processed_at). Without a
# composite index both scan every historical row, so claim latency grows with
# total table size rather than with the number of pending events.
Index(
    "ix_outbox_claim",
    OutboxEvent.status,
    OutboxEvent.attempts,
    OutboxEvent.claimed_at,
    OutboxEvent.created_at,
)
Index("ix_outbox_retention", OutboxEvent.status, OutboxEvent.processed_at)
Index("ix_prediction_events_retention", PredictionEvent.created_at, PredictionEvent.label)


def init_db(force: bool = False) -> None:
    """
    Ensure the event store schema exists.

    Idempotent and cached: this used to run ``create_all`` — a set of reflection
    queries — on *every* prediction, every worker poll and every monitoring read,
    adding database round-trips to the request hot path.

    Note that ``create_all`` only ever creates missing tables; it cannot alter
    existing ones. Schema changes go through Alembic (see ``alembic/``).
    """
    global _SCHEMA_READY
    if _SCHEMA_READY and not force:
        return

    # Ensure SQLite file parent exists (CI and fresh environments)
    if ENGINE.url.get_backend_name() == "sqlite":
        db_file = ENGINE.url.database
        if db_file and db_file not in {":memory:", ""}:
            db_path = Path(db_file)
            if not db_path.is_absolute():
                db_path = Path.cwd() / db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=ENGINE)
    _SCHEMA_READY = True


def now_utc() -> datetime:
    return datetime.now(timezone.utc)

