"""Infer per-column types for API / metadata from a feature DataFrame."""

from __future__ import annotations

import pandas as pd


def infer_feature_types(df: pd.DataFrame) -> dict[str, str]:
    """
    Map column name -> 'int' | 'float' | 'str' | 'bool' for Pydantic + metadata.

    Uses pandas dtypes; object columns become 'str'.
    """
    out: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = "bool"
        elif pd.api.types.is_integer_dtype(s):
            out[col] = "int"
        elif pd.api.types.is_float_dtype(s):
            out[col] = "float"
        else:
            out[col] = "str"
    return out
