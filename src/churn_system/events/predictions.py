"""Prediction event storage with retry-on-failure for DB writes."""

from __future__ import annotations

import hashlib
import hmac
import os
import unicodedata
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


SUBJECT_SALT_ENV = "CHURN_SUBJECT_KEY_SALT"
ALLOW_UNSALTED_ENV = "CHURN_ALLOW_UNSALTED_SUBJECT_KEYS"


class SubjectSaltMissingError(RuntimeError):
    """Raised when pseudonymisation is requested without a configured salt."""


def _subject_salt() -> bytes | None:
    """
    Return the configured salt, or None when pseudonymisation is explicitly disabled.

    There is deliberately NO default. The previous hardcoded "churn-default-salt"
    was published in this open-source repository, and customer identifiers are a
    small enumerable space — so the entire subject_key column was reversible by
    brute force in seconds, defeating the only privacy property the event store
    claimed. An unset salt now fails closed unless disabling pseudonymisation is an
    explicit, deliberate choice.
    """
    salt = os.environ.get(SUBJECT_SALT_ENV, "").strip()
    if salt:
        return salt.encode("utf-8")

    if os.environ.get(ALLOW_UNSALTED_ENV, "").strip() == "1":
        return None

    raise SubjectSaltMissingError(
        f"{SUBJECT_SALT_ENV} is not set. Generate one with "
        "`python -c \"import secrets; print(secrets.token_hex(32))\"`, or set "
        f"{ALLOW_UNSALTED_ENV}=1 to run without storing subject keys at all."
    )


def subject_key(subject_id: str | None) -> str | None:
    """
    Derive a stable pseudonymous key for a customer identifier.

    Stored instead of the identifier itself, so prediction history can be joined to
    a customer for labelling or erasure without the event store holding directly
    identifying data.

    The identifier is Unicode-normalised (NFC) before hashing. Without this, the
    same name submitted as NFC at prediction time and NFD at erasure time hashes
    differently — so a GDPR erasure request silently matched nothing while
    reporting success.
    """
    if not subject_id or not subject_id.strip():
        return None

    salt = _subject_salt()
    if salt is None:
        return None  # pseudonymisation explicitly disabled

    normalised = unicodedata.normalize("NFC", subject_id.strip())
    return hmac.new(salt, normalised.encode("utf-8"), hashlib.sha256).hexdigest()[:64]


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
