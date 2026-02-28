import shutil
import json
from pathlib import Path

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG
from churn_system.lifecycle.lineage import record_lineage

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])


def promote_model(version: str):
    """
    Promote a trained model version to production.

    Parameters
        version : str
            Name of the model directory inside `models/experiments/`.
            Example: "churn_model_v1"

        1. Checks whether the requested experiment exists.
        2. Removes the existing production model (if any).
        3. Copies the selected experiment info for production.
        4. Makes the promoted model the one used by the API.

        Note :
            if the requested model version does not exist it raises ValueError.

        The API Always loads models from the production directory, never directly from experiments.
    """

    experiments_dir = Path(CONFIG["paths"]["experiments_dir"])
    production_dir = Path(CONFIG["paths"]["production_model"]).parent

    source = experiments_dir / version
    target = production_dir

    if not source.exists():
        raise ValueError(f"Model version {version} does not exist.")

    metadata_path = source / "metadata.json"

    if not metadata_path.exists():
        raise ValueError("metadata.json missing for experiment.")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    parent_model = None
    existing_metadata = target / "metadata.json"

    if existing_metadata.exists():
        try:
            with open(existing_metadata, "r") as f:
                parent_data = json.load(f)
                parent_model = parent_data.get("model_version")
        except Exception:
            parent_model = "unknown"

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(source, target)

    logger.info(f"Model {version} promoted to production.")

    record_lineage(
        model_version=version,
        metrics=metadata.get("metrics", {}),
        dataset_used=metadata.get("dataset", "unknown"),
        trigger="drift_retraining",
        parent_model=parent_model,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m churn_system.lifecycle.promote <version>")
        sys.exit(1)

    promote_model(sys.argv[1])
