"""Prediction event storage with retry-on-failure for DB writes."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import OperationalError

from churn_system.events.db import OutboxEvent, PredictionEvent, SessionLocal, init_db, now_utc
from churn_system.inference.model_contract import load_model_contract
from churn_system.utils.retry import retry_with_backoff

SENSITIVE_KEYS = frozenset(
    {
        "CustomerID",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
    }
)


def _redact(features: dict[str, Any]) -> dict[str, Any]:
    """Strip PII / geo fields before durable storage."""
    return {k: v for k, v in features.items() if k not in SENSITIVE_KEYS}


def subject_key(subject_id: str | None) -> str | None:
    """
    Derive a stable pseudonymous key for a customer identifier.

    Stored instead of the identifier itself, so prediction history can be joined to
    a customer for labelling or erasure without the event store ever holding
    directly identifying data. Salted with CHURN_SUBJECT_KEY_SALT to stop the hash
    from being reversed with a dictionary of plausible ids.
    """
    if not subject_id or not subject_id.strip():
        return None
    salt = os.environ.get("CHURN_SUBJECT_KEY_SALT", "churn-default-salt").encode()
    return hmac.new(salt, subject_id.strip().encode(), hashlib.sha256).hexdigest()[:64]


def store_prediction_event(
    *,
    request_id: str,
    raw_features: dict[str, Any],
    probability: float,
    prediction: int,
    latency_seconds: float,
    subject_id: str | None = None,
) -> None:
    """
    Store a durable, redacted prediction event + emit an outbox message.

    Retries up to 3 times with exponential backoff on transient DB errors.
    """
    init_db()
    meta = load_model_contract()
    model_version = meta.get("model_version")

    redacted = _redact(raw_features)
    subject = subject_key(subject_id)

    def _write():
        with SessionLocal() as session:
            session.add(
                PredictionEvent(
                    request_id=request_id,
                    created_at=now_utc(),
                    model_version=model_version,
                    probability=float(probability),
                    prediction=int(prediction),
                    latency_seconds=float(latency_seconds),
                    features=redacted,
                    subject_key=subject,
                )
            )
            session.add(
                OutboxEvent(
                    created_at=now_utc(),
                    event_type="prediction_made",
                    payload={
                        "request_id": request_id,
                        "model_version": model_version,
                        "probability": float(probability),
                        "prediction": int(prediction),
                    },
                    processed_at=None,
                )
            )
            session.commit()

    retry_with_backoff(
        _write,
        max_retries=3,
        base_delay=0.3,
        retryable_exceptions=(OperationalError, OSError),
        operation_name="store_prediction_event",
    )


def record_label(request_id: str, label: int) -> bool:
    """
    Attach ground truth to a previously served prediction.

    This is what turns monitoring from "the inputs look different" into "the model
    is wrong": without labels, model health can only ever measure input drift.

    Returns True when a matching prediction was found and updated.
    """
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")

    init_db()

    with SessionLocal() as session:
        result = session.execute(
            update(PredictionEvent)
            .where(PredictionEvent.request_id == request_id)
            .values(label=int(label), labeled_at=now_utc())
        )
        session.add(
            OutboxEvent(
                created_at=now_utc(),
                event_type="prediction_labeled",
                payload={"request_id": request_id, "label": int(label)},
                processed_at=None,
            )
        )
        session.commit()

    return bool(result.rowcount)


def load_labeled_events(limit: int | None = None) -> list[PredictionEvent]:
    """Return prediction events that have ground truth attached."""
    init_db()
    with SessionLocal() as session:
        stmt = (
            select(PredictionEvent)
            .where(PredictionEvent.label.is_not(None))
            .order_by(PredictionEvent.created_at.asc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(session.execute(stmt).scalars().all())


def purge_subject(subject_id: str) -> int:
    """
    Delete every stored prediction for a customer (GDPR erasure).

    Matches on the pseudonymous key, so the caller supplies the original identifier
    and nothing identifying is ever compared or stored. Returns the row count.
    """
    key = subject_key(subject_id)
    if key is None:
        return 0

    init_db()
    with SessionLocal() as session:
        result = session.execute(
            delete(PredictionEvent).where(PredictionEvent.subject_key == key)
        )
        session.commit()

    return int(result.rowcount or 0)
