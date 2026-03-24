"""
Automatic API Schema Generator

Builds FastAPI request schema dynamically from production model metadata
with typed fields (not Any).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ConfigDict, Field, create_model

from churn_system.config.config import CONFIG
from churn_system.training.feature_types import infer_feature_types


def _load_metadata() -> dict[str, Any]:
    metadata_path = (
        Path(CONFIG["paths"]["production_model"]).parent / "metadata.json"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Production metadata not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_feature_schema() -> list[str]:
    return load_model_metadata()["feature_schema"]


def load_model_metadata() -> dict[str, Any]:
    return _load_metadata()


def _load_feature_types_from_reference(
    feature_schema: list[str],
) -> dict[str, str]:
    ref_path = Path(CONFIG["paths"]["training_reference"])
    if not ref_path.exists():
        return {c: "str" for c in feature_schema}
    df = pd.read_csv(ref_path, nrows=512)
    missing = [c for c in feature_schema if c not in df.columns]
    if missing:
        return {c: "str" for c in feature_schema}
    subset = df[feature_schema]
    return infer_feature_types(subset)


def load_feature_types() -> dict[str, str]:
    meta = _load_metadata()
    feature_schema: list[str] = meta["feature_schema"]
    if "feature_types" in meta and isinstance(meta["feature_types"], dict):
        ft = meta["feature_types"]
        # Ensure every column has a type
        out = {c: ft.get(c, "str") for c in feature_schema}
        return out
    return _load_feature_types_from_reference(feature_schema)


def _python_type_for(name: str) -> type:
    mapping = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    return mapping.get(name, str)


def generate_request_model():
    """
    Dynamically create a Pydantic request model with typed fields.
    """
    features = load_feature_schema()
    types_map = load_feature_types()

    fields: dict[str, Any] = {}
    for feature in features:
        tname = types_map.get(feature, "str")
        py_t = _python_type_for(tname)
        fields[feature] = (py_t, Field(..., description=f"Feature {feature}"))

    RequestModel = create_model(
        "DynamicPredictionRequest",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )

    return RequestModel
