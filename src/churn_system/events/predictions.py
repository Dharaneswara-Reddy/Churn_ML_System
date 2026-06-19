"""Prediction event storage with retry-on-failure for DB writes."""

from __future__ import annotations

from typing import Any

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


def store_prediction_event(
    *,
    request_id: str,
    raw_features: dict[str, Any],
    probability: float,
    prediction: int,
    latency_seconds: float,
) -> None:
    """
    Store a durable, redacted prediction event + emit an outbox message.

    Retries up to 3 times with exponential backoff on transient DB errors.
    """
    init_db()
    meta = load_model_contract()
    model_version = meta.get("model_version")

    redacted = _redact(raw_features)

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
