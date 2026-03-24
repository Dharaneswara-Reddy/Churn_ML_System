from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.pandas import Check, Column, DataFrameSchema

_TYPE_MAP: dict[str, Any] = {
    "str": pa.String,
    "int": pa.Int,
    "float": pa.Float,
    "bool": pa.Bool,
}


def load_schema_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_pandera_schema(schema_dict: dict[str, Any]) -> DataFrameSchema:
    columns = {}
    for name, spec in schema_dict.get("columns", {}).items():
        t = _TYPE_MAP.get(str(spec.get("type", "str")), pa.String)
        required = bool(spec.get("required", True))
        nullable = bool(spec.get("nullable", False))
        columns[name] = Column(t, required=required, nullable=nullable)

    checks = []
    for col, c in (schema_dict.get("checks") or {}).items():
        if "allowed" in c:
            allowed = set(c["allowed"])
            checks.append(Check.isin(allowed),)

    # Column-specific checks
    col_checks = {}
    for col, c in (schema_dict.get("checks") or {}).items():
        col_checks[col] = []
        if "allowed" in c:
            allowed = set(c["allowed"])
            col_checks[col].append(Check.isin(allowed))

    for col, cks in col_checks.items():
        if col in columns:
            columns[col] = Column(
                columns[col].dtype,
                required=columns[col].required,
                nullable=columns[col].nullable,
                checks=cks,
            )

    return DataFrameSchema(columns, coerce=True, strict=False)


def validate_dataframe(df: pd.DataFrame, *, schema_path: Path) -> pd.DataFrame:
    schema_dict = load_schema_yaml(schema_path)
    schema = build_pandera_schema(schema_dict)
    return schema.validate(df, lazy=True)

