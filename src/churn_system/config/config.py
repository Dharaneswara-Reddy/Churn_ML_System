from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent / "settings.yaml"

PATH_ENV_OVERRIDES = {
    "raw_data": "CHURN_RAW_DATA_PATH",
    "retraining_data": "CHURN_RETRAINING_DATA_PATH",
    "training_reference": "CHURN_TRAINING_REFERENCE_PATH",
    "production_model": "CHURN_PRODUCTION_MODEL_PATH",
    "experiments_dir": "CHURN_EXPERIMENTS_DIR",
    "monitoring_dir": "CHURN_MONITORING_DIR",
    "lineage_path": "CHURN_LINEAGE_PATH",
    "prediction_log_csv": "CHURN_PREDICTION_LOG_CSV",
}


def _set_if_env(cfg: dict[str, Any], section: str, key: str, env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        cfg.setdefault(section, {})[key] = value


def _set_float_if_env(cfg: dict[str, Any], section: str, key: str, env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        cfg.setdefault(section, {})[key] = float(value)


def _set_int_if_env(cfg: dict[str, Any], section: str, key: str, env_name: str) -> None:
    value = os.environ.get(env_name)
    if value is not None and value.strip():
        cfg.setdefault(section, {})[key] = int(value)


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # CI / containers: override paths without editing YAML
    paths = cfg.setdefault("paths", {})
    for key, env_name in PATH_ENV_OVERRIDES.items():
        if p := os.environ.get(env_name):
            paths[key] = p

    _set_float_if_env(cfg, "inference", "threshold", "CHURN_INFERENCE_THRESHOLD")
    _set_if_env(cfg, "api", "rate_limit", "CHURN_API_RATE_LIMIT")
    _set_if_env(cfg, "event_store", "database_url", "CHURN_EVENT_STORE_DATABASE_URL")
    _set_if_env(cfg, "mlflow", "tracking_uri", "CHURN_MLFLOW_TRACKING_URI")
    _set_int_if_env(cfg, "scheduler", "interval_seconds", "CHURN_SCHEDULER_INTERVAL_SECONDS")
    _set_int_if_env(cfg, "training", "min_rows", "CHURN_TRAINING_MIN_ROWS")
    _set_if_env(cfg, "training", "selection_metric", "CHURN_TRAINING_SELECTION_METRIC")
    _set_if_env(cfg, "model_promotion", "metric", "CHURN_MODEL_PROMOTION_METRIC")
    _set_float_if_env(
        cfg,
        "model_promotion",
        "min_improvement",
        "CHURN_MODEL_PROMOTION_MIN_IMPROVEMENT",
    )

    return cfg


CONFIG = load_config()
