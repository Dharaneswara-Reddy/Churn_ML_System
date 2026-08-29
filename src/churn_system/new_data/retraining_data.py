"""
Build the retraining dataset by combining the original training data with
*labelled* production traffic.

Only predictions that have ground truth attached are usable. A prediction on its
own carries no target column, so the previous implementation — which tried to
merge raw prediction rows and required an exact column match — could never add a
single row: the target is rejected at inference by design, and eight more fields
are stripped by PII redaction. Every "retrain" therefore re-fit the same static
CSV and produced a bit-identical model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.features.build_features import TARGET_COLUMN
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["training"])


def _raw_data_path() -> Path:
    return Path(CONFIG["paths"]["raw_data"])


def _output_path() -> Path:
    return Path(CONFIG["paths"]["retraining_data"])


def labelled_production_frame() -> pd.DataFrame:
    """
    Return labelled production predictions as a training-shaped frame.

    The stored feature payload is already redacted, so the result has fewer columns
    than the raw dataset; missing columns are filled by the caller so the union can
    be concatenated.
    """
    from churn_system.events.predictions import load_labeled_events

    events = load_labeled_events()
    if not events:
        return pd.DataFrame()

    records = []
    for event in events:
        row = dict(event.features)
        row[TARGET_COLUMN] = int(event.label)
        records.append(row)

    return pd.DataFrame.from_records(records)


def _passes_training_contract(frame: pd.DataFrame) -> bool:
    """
    Check a candidate retraining frame against the training validation contract.

    Uses the exact function the training pipeline runs, on a copy, so this can
    never disagree with the gate it is protecting.
    """
    from churn_system.training.steps.data_validation import run_data_validation

    try:
        run_data_validation(frame.copy())
    except Exception:
        return False
    return True


def build_retraining_dataset() -> Path:
    """
    Write the retraining dataset and return its path.

    Falls back to the original data alone when no labelled production rows exist —
    which is the honest behaviour, and is now visible in the logs rather than
    silently pretending fresh data was incorporated.
    """
    raw_path = _raw_data_path()
    if not raw_path.exists():
        raise FileNotFoundError(f"Original dataset missing: {raw_path}")

    base_df = pd.read_csv(raw_path)
    production_df = labelled_production_frame()

    if production_df.empty:
        logger.info(
            "No labelled production data available — retraining on the base dataset "
            "only. Collect outcomes via POST /feedback/{request_id} to improve this."
        )
        combined = base_df
    else:
        usable = [c for c in production_df.columns if c in base_df.columns]
        missing = [c for c in base_df.columns if c not in production_df.columns]

        aligned = production_df[usable].copy()
        for column in missing:
            aligned[column] = pd.NA
        aligned = aligned[base_df.columns]

        combined = pd.concat([base_df, aligned], ignore_index=True)

        # Validate against the *same* contract training will apply, before this
        # file is written. Ingestion prefers the retraining dataset whenever it
        # exists, so writing a frame that fails validation does not merely skip
        # one cycle — it permanently breaks every subsequent training run, and
        # the scheduler swallows the exception so the failure is silent.
        if not _passes_training_contract(combined):
            logger.error(
                "Merged retraining dataset failed the training contract; falling "
                "back to the base dataset. Labelled production rows are missing "
                "%d column(s) that validation requires (%s). These are stripped by "
                "PII redaction and cannot be recovered — drop them from the feature "
                "set, or relax their nullability, to make labelled rows usable.",
                len(missing),
                ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""),
            )
            combined = base_df
        else:
            logger.info(
                "Added %d labelled production samples (%d columns carried over, "
                "%d filled as missing).",
                len(aligned),
                len(usable),
                len(missing),
            )

    output = _output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    logger.info("Retraining dataset written to %s (%d rows)", output, len(combined))

    return output
