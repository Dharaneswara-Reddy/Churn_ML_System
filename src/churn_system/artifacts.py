"""Helpers for model artifact paths and bundle validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from churn_system.config.config import CONFIG


def _cfg(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if config is not None else CONFIG


def production_model_path(config: dict[str, Any] | None = None) -> Path:
    return Path(_cfg(config)["paths"]["production_model"])


def production_model_dir(config: dict[str, Any] | None = None) -> Path:
    return production_model_path(config).parent


def production_metadata_path(config: dict[str, Any] | None = None) -> Path:
    return production_model_dir(config) / "metadata.json"


def experiments_dir(config: dict[str, Any] | None = None) -> Path:
    return Path(_cfg(config)["paths"]["experiments_dir"])


def experiment_dir(version: str, config: dict[str, Any] | None = None) -> Path:
    return experiments_dir(config) / version


def latest_experiment_dir(config: dict[str, Any] | None = None) -> Path | None:
    import re
    pattern = re.compile(r"^churn_model_\d{8}_\d{6}$")
    versions = sorted([
        d for d in experiments_dir(config).glob("churn_model_*")
        if d.is_dir() and pattern.match(d.name) and (d / "metadata.json").exists()
    ])
    return versions[-1] if versions else None


def metadata_path_for_model(model_path: Path) -> Path:
    return model_path.parent / "metadata.json"


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata must be a JSON object: {metadata_path}")
    return data


def validate_model_bundle(
    model_path: Path,
    *,
    metadata_path: Path | None = None,
    require_model: bool = True,
) -> dict[str, Any]:
    """
    Validate the serving contract for a model artifact bundle.

    A deployable bundle is a model pickle plus sibling metadata.json. Metadata
    must carry a non-empty, ordered feature schema because serving depends on it.
    """

    if require_model and not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    resolved_metadata_path = metadata_path or metadata_path_for_model(model_path)
    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {resolved_metadata_path}")

    metadata = load_metadata(resolved_metadata_path)
    feature_schema = metadata.get("feature_schema")
    if not isinstance(feature_schema, list) or not feature_schema:
        raise ValueError("metadata.json must contain a non-empty feature_schema list")
    if not all(isinstance(feature, str) and feature for feature in feature_schema):
        raise ValueError("feature_schema entries must be non-empty strings")

    feature_count = metadata.get("feature_count")
    if feature_count is not None and int(feature_count) != len(feature_schema):
        raise ValueError(
            "metadata feature_count does not match feature_schema length "
            f"({feature_count} != {len(feature_schema)})"
        )

    metrics = metadata.get("metrics", {})
    if metrics is not None and not isinstance(metrics, dict):
        raise ValueError("metadata metrics must be an object when present")

    return metadata
