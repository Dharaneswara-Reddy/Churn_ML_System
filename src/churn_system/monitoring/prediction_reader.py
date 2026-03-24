from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from churn_system.events.db import PredictionEvent, SessionLocal, init_db


def load_predictions_df(limit: int | None = None) -> pd.DataFrame:
    """
    Load prediction events into a DataFrame for monitoring jobs.
    """
    init_db()
    with SessionLocal() as session:
        stmt = select(PredictionEvent).order_by(PredictionEvent.created_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = session.execute(stmt).scalars().all()

    # Oldest -> newest for time-series reporting
    rows = list(reversed(rows))
    if not rows:
        return pd.DataFrame()

    # Flatten features dict into columns (redacted features only)
    records: list[dict] = []
    for r in rows:
        rec = dict(r.features)
        rec.update(
            {
                "request_id": r.request_id,
                "timestamp": r.created_at.isoformat(),
                "prediction_probability": r.probability,
                "prediction": r.prediction,
                "latency_seconds": r.latency_seconds,
                "model_version": r.model_version,
            }
        )
        records.append(rec)

    return pd.DataFrame.from_records(records)

