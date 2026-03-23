import json
import shutil
from pathlib import Path

from churn_system.config.config import CONFIG
from churn_system.lifecycle.lineage import record_lineage
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])


def load_metadata(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def schemas_match(prod_meta, new_meta):
    prod_schema = prod_meta.get("feature_schema", [])
    new_schema = new_meta.get("feature_schema", [])

    return prod_schema == new_schema


def promote_model(version: str):
    """
    Promote a trained model version to production.

    Ensures schema compatibility before promotion.
    """

    experiments_dir = Path(CONFIG["paths"]["experiments_dir"])
    production_dir = Path(CONFIG["paths"]["production_model"]).parent

    source = experiments_dir / version
    target = production_dir / "current"

    if not source.exists():
        raise ValueError(f"Model version {version} does not exist.")

    new_metadata_path = source / "metadata.json"

    if not new_metadata_path.exists():
        raise ValueError("metadata.json missing for experiment.")

    new_metadata = load_metadata(new_metadata_path)

    parent_model = None
    existing_metadata_path = target / "metadata.json"

    # ✅ Schema Safety Check
    if existing_metadata_path.exists():
        prod_metadata = load_metadata(existing_metadata_path)

        parent_model = prod_metadata.get("model_version")

        if not schemas_match(prod_metadata, new_metadata):
            logger.error(
                "Feature schema mismatch detected. Promotion blocked."
            )
            logger.error("Production and challenger schemas differ.")
            return

    # Promote
    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(source, target)

    logger.info(f"Model {version} promoted to production.")

    record_lineage(
        model_version=version,
        metrics=new_metadata.get("metrics", {}),
        dataset_used=new_metadata.get("dataset", "unknown"),
        trigger="drift_retraining",
        parent_model=parent_model,
    )
