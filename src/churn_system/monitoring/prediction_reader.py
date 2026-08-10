from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import select

from churn_system.config.config import CONFIG
from churn_system.events.db import PredictionEvent, SessionLocal, init_db


def load_reference_and_production() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Load the training reference and the production prediction feed for monitoring.

    Returns None when either side is unavailable, so callers can report "unknown"
    rather than guessing. Falls back to the legacy prediction CSV only when the
    event store is empty.
    """
    reference_path = Path(CONFIG["paths"]["training_reference"])
    legacy_path = Path(CONFIG["paths"]["prediction_log_csv"])

    if not reference_path.exists():
        return None

    reference_df = pd.read_csv(reference_path)
    production_df = load_predictions_df()

    if production_df.empty:
        if not legacy_path.exists():
            return None
        production_df = pd.read_csv(legacy_path)

    if production_df.empty:
        return None

    return reference_df, production_df


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

