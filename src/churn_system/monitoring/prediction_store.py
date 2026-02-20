import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["monitoring"])

LOG_PATH = Path("data/inference_logs/predictions.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def store_prediction(input_record: dict, probability: float, prediction: int):
    """
    Store inference request safely with fixed schema.
    """

    record = input_record.copy()


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
        index=False
    )

    logger.info("Prediction stored successfully.")
