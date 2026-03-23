import os
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "settings.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # CI / containers: override paths without editing YAML
    paths = cfg.setdefault("paths", {})
    if p := os.environ.get("CHURN_RAW_DATA_PATH"):
        paths["raw_data"] = p
    if p := os.environ.get("CHURN_EXPERIMENTS_DIR"):
        paths["experiments_dir"] = p
    if p := os.environ.get("CHURN_TRAINING_REFERENCE_PATH"):
        paths["training_reference"] = p
    if p := os.environ.get("CHURN_PRODUCTION_MODEL_PATH"):
        paths["production_model"] = p

    return cfg


CONFIG = load_config()
