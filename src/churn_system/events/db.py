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


def init_db() -> None:
    # Ensure SQLite file parent exists (CI and fresh environments)
    if ENGINE.url.get_backend_name() == "sqlite":
        db_file = ENGINE.url.database
        if db_file and db_file not in {":memory:", ""}:
            db_path = Path(db_file)
            if not db_path.is_absolute():
                db_path = Path.cwd() / db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=ENGINE)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)

