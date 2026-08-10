"""
Durable event storage (SQLite by default).

This replaces CSV append logging for inference events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, MetaData, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from churn_system.config.config import load_config


def _db_url() -> str:
    cfg = load_config()
    return str(cfg.get("event_store", {}).get("database_url", "sqlite:///./data/churn_events.db"))


ENGINE = create_engine(_db_url(), future=True)
SessionLocal = sessionmaker(bind=ENGINE, autocommit=False, autoflush=False, future=True)


class Base(DeclarativeBase):
    metadata = MetaData()


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
    features: Mapped[dict[str, Any]] = mapped_column(JSON)

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
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
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


_SCHEMA_READY = False


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

