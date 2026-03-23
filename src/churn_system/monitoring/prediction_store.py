from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["monitoring"])

LOG_PATH = Path("data/inference_logs/predictions.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Do not persist raw identifiers / location fields suitable for minimization.
_SENSITIVE_KEYS = frozenset(
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


def _redact_record(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _SENSITIVE_KEYS}


def store_prediction(
    input_record: dict,
    probability: float,
    prediction: int,
    *,
    request_id: str,
) -> None:
    """
    Persist a minimal, redacted row for monitoring (no raw PII / location).
    """

    record = _redact_record(input_record.copy())
    record["request_id"] = request_id
    record["prediction_probability"] = float(probability)
    record["prediction"] = int(prediction)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()

    df = pd.DataFrame([record])
    df = df.reindex(sorted(df.columns), axis=1)

    write_header = not LOG_PATH.exists()

    df.to_csv(
        LOG_PATH,
        mode="a",
        header=write_header,
        index=False,
    )

    logger.info("Prediction stored | request_id=%s", request_id)
