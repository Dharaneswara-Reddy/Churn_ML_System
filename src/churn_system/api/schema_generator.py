"""
Automatic API Schema Generator

Builds the FastAPI request schema dynamically from production model metadata,
with typed fields (not ``Any``).

Backward compatibility across a feature change
----------------------------------------------
The request schema is derived from the champion's ``feature_schema``, so removing
a feature from the model removes a field from the API. With ``extra="forbid"``
that turns every previously valid request into a 422 the moment a new champion is
promoted — a breaking change delivered by a background scheduler rather than by a
deploy, which is the worst possible way to ship one.

Fields the model no longer consumes are therefore accepted and ignored rather than
rejected. The set is derived, not hardcoded: it is the geographic columns
``build_features`` drops, minus whatever the current champion still uses, so it
retires itself automatically as models change. Each accepted-and-ignored field is
counted on ``churn_deprecated_request_fields_total`` so the retirement date can be
driven by evidence that no caller still sends them.

Leakage columns are pointedly *not* included even though ``build_features`` drops
them too. They were never fields of any published request schema, so no client has
them to send, and accepting them would widen the public API to take ``Churn
Label`` and ``Churn Score`` — the target restated. A caller posting those is
posting training data to an inference endpoint, and deserves a 422 saying so.

``extra="forbid"`` still applies to everything else: a misspelled feature name is
a 422, not a silently ignored field. Strict mode
(``CHURN_STRICT_REQUEST_SCHEMA=1``) drops the shim entirely for deployments that
would rather break loudly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ConfigDict, Field, create_model

from churn_system.config.config import CONFIG
from churn_system.features.build_features import GEOGRAPHIC_COLUMNS
from churn_system.inference.model_contract import load_model_contract
from churn_system.logging.logger import get_logger
from churn_system.training.feature_types import infer_feature_types

logger = get_logger(__name__, CONFIG["logging"]["api"])

STRICT_SCHEMA_ENV = "CHURN_STRICT_REQUEST_SCHEMA"


def _load_metadata() -> dict[str, Any]:
    return load_model_contract()


def load_feature_schema() -> list[str]:
    return load_model_metadata()["feature_schema"]


def load_model_metadata() -> dict[str, Any]:
    return _load_metadata()


def _load_feature_types_from_reference(
    feature_schema: list[str],
) -> dict[str, str]:
    ref_path = Path(CONFIG["paths"]["training_reference"])
    if not ref_path.exists():
        return dict.fromkeys(feature_schema, "str")
    df = pd.read_csv(ref_path, nrows=512)
    missing = [c for c in feature_schema if c not in df.columns]
    if missing:
        return dict.fromkeys(feature_schema, "str")
    subset = df[feature_schema]
    return infer_feature_types(subset)


def load_feature_types() -> dict[str, str]:
    meta = _load_metadata()
    feature_schema: list[str] = meta["feature_schema"]
    if "feature_types" in meta and isinstance(meta["feature_types"], dict):
        ft = meta["feature_types"]
        # Ensure every column has a type
        return {c: ft.get(c, "str") for c in feature_schema}
    return _load_feature_types_from_reference(feature_schema)


def _python_type_for(name: str) -> type:
    mapping = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    return mapping.get(name, str)


def strict_schema_enabled() -> bool:
    """Read at call time so a test can flip it without reimporting the module."""
    return os.environ.get(STRICT_SCHEMA_ENV, "0").strip().lower() in {"1", "true", "yes"}


def deprecated_request_fields() -> list[str]:
    """
    Fields accepted for backward compatibility and then ignored.

    Derived rather than enumerated: the geographic columns were required by the
    previously published request schema and are dropped by ``build_features``
    anyway, so accepting them costs nothing. Anything the current champion still
    uses is excluded, which means a feature that comes back into the model stops
    being deprecated on its own and becomes required again.

    Leakage columns and the target are deliberately absent — see this module's
    docstring.
    """
    if strict_schema_enabled():
        return []

    try:
        active = set(load_feature_schema())
    except Exception:
        # Without a champion there is no schema to be compatible with.
        return []

    return [column for column in GEOGRAPHIC_COLUMNS if column not in active]


def generate_request_model():
    """
    Dynamically create a Pydantic request model with typed fields.

    Active features stay required and typed. Deprecated fields are optional,
    untyped and default to ``None``, so an old caller's payload validates
    unchanged while a new caller may simply omit them.
    """
    features = load_feature_schema()
    types_map = load_feature_types()

    fields: dict[str, Any] = {}
    for feature in features:
        tname = types_map.get(feature, "str")
        py_t = _python_type_for(tname)
        fields[feature] = (py_t, Field(..., description=f"Feature {feature}"))

    deprecated = deprecated_request_fields()
    for column in deprecated:
        # Deliberately `Any`: the point is to accept whatever an existing client
        # already sends, including types the old schema coerced differently.
        fields[column] = (
            Any,
            Field(
                default=None,
                deprecated=True,
                description=(
                    f"Deprecated. '{column}' is no longer used by the model and is "
                    "ignored. Accepted so existing clients keep working; remove it "
                    "from new integrations."
                ),
            ),
        )

    if deprecated:
        logger.info(
            "Request schema accepts %d deprecated field(s) for backward "
            "compatibility: %s",
            len(deprecated),
            ", ".join(deprecated),
        )

    return create_model(
        "DynamicPredictionRequest",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
