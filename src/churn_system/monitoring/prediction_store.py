"""
Prediction storage adapter.

P1: durable storage in DB outbox (replaces CSV append).
"""

from __future__ import annotations

from typing import Any

from churn_system.events.predictions import store_prediction_event


def store_prediction(
    input_record: dict[str, Any],
    probability: float,
    prediction: int,
    *,
    request_id: str,
) -> None:
    store_prediction_event(
        request_id=request_id,
        raw_features=input_record,
        probability=probability,
        prediction=prediction,
        latency_seconds=0.0,
    )
