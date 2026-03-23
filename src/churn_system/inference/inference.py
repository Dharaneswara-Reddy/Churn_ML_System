"""
Offline / library inference helpers (not the FastAPI layer).

Used by inference_pipeline and ad-hoc scripts.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.features.build_features import build_features
from churn_system.schema import validate_inference_data


def _load_model():
    model_path = Path(CONFIG["paths"]["production_model"])
    with open(model_path, "rb") as f:
        return pickle.load(f)


def run_inference(
    payload: dict[str, Any],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Run churn prediction on one raw feature row (same keys as API body).
    """
    from churn_system.config.config import load_config

    cfg = load_config()
    thr = (
        threshold
        if threshold is not None
        else float(cfg["inference"]["threshold"])
    )

    df = pd.DataFrame([payload])
    df = build_features(df, training=False)
    df_valid = validate_inference_data(df)

    model = _load_model()
    prob = float(model.predict_proba(df_valid)[:, 1][0])
    pred = int(prob >= thr)

    return {
        "churn_probability": prob,
        "prediction": pred,
        "threshold": thr,
    }
