"""
SHAP-based Explainable AI Engine.

Provides local (per-prediction) and global (model-wide) explanations
using SHAP (SHapley Additive exPlanations) values.

Architecture:
  - Uses a TreeExplainer for tree-based models (RandomForest, GradientBoosting)
    which computes exact Shapley values in polynomial time.
  - Falls back to KernelExplainer for linear models (LogisticRegression)
    which uses a weighted linear regression approximation.
  - A background dataset sample (100 rows from training data) is cached
    once and reused across all explanation requests to avoid recomputation.

Thread Safety:
  - The explainer is lazily initialized and cached behind a threading.Lock
    to prevent race conditions when multiple API threads request
    explanations concurrently.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from churn_system.config.config import CONFIG
from churn_system.features.build_features import build_features
from churn_system.logging.logger import get_logger
from churn_system.schema import validate_inference_data
from churn_system.serving.model_registry import ModelRegistry

logger = get_logger(__name__, CONFIG["logging"].get("explainability", "explainability.log"))

# ── Cached state ─────────────────────────────────────────────────────────────
_explainer: shap.Explainer | None = None
_explainer_lock = threading.Lock()
_background_data: pd.DataFrame | None = None
_feature_names: list[str] | None = None

# Number of background samples for KernelExplainer (higher = more accurate
# but slower). TreeExplainer doesn't need this.
BACKGROUND_SAMPLE_SIZE = 100


def _load_background_data() -> pd.DataFrame:
    """
    Load and cache a reference sample from training data.

    This sample is used by KernelExplainer as the background dataset
    to integrate over when computing Shapley values.
    """
    global _background_data, _feature_names

    if _background_data is not None:
        return _background_data

    train_path = Path(CONFIG["paths"]["training_reference"])
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training reference data not found at {train_path}. "
            "Run a training pipeline first to generate reference data."
        )

    df = pd.read_csv(train_path)
    df = build_features(df, training=False)
    df = validate_inference_data(df)

    # Take a stratified sample to keep the background set manageable
    if len(df) > BACKGROUND_SAMPLE_SIZE:
        df = df.sample(n=BACKGROUND_SAMPLE_SIZE, random_state=42)

    _background_data = df
    _feature_names = list(df.columns)
    logger.info(
        "Background data loaded | samples=%d | features=%d",
        len(df),
        len(df.columns),
    )
    return _background_data


def _get_explainer() -> shap.Explainer:
    """
    Lazily initialize and cache the SHAP explainer.

    Uses double-checked locking to avoid holding the lock on the hot path
    once the explainer is initialized.
    """
    global _explainer

    if _explainer is not None:
        return _explainer

    with _explainer_lock:
        if _explainer is not None:
            return _explainer

        model = ModelRegistry.instance().get_model()
        background = _load_background_data()

        # Detect model type from the pipeline's final estimator
        final_estimator = model.named_steps.get("model", model)
        estimator_name = type(final_estimator).__name__

        if estimator_name in ("RandomForestClassifier", "GradientBoostingClassifier"):
            _explainer = shap.TreeExplainer(model, data=background)
            logger.info("TreeExplainer initialized for %s", estimator_name)
        else:
            # KernelExplainer works for any model but is slower
            summary = shap.kmeans(background, min(10, len(background)))
            _explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(pd.DataFrame(x, columns=background.columns)),
                summary,
            )
            logger.info("KernelExplainer initialized for %s", estimator_name)

        return _explainer


def reset_explainer() -> None:
    """Reset the cached explainer (for testing or after model hot-reload)."""
    global _explainer, _background_data, _feature_names
    with _explainer_lock:
        _explainer = None
        _background_data = None
        _feature_names = None


def explain_prediction(raw_features: dict) -> dict[str, Any]:
    """
    Generate a SHAP explanation for a single prediction.

    Parameters
    ----------
    raw_features : dict
        Raw input features (same format as the /predict endpoint payload).

    Returns
    -------
    dict
        Contains:
        - ``shap_values``: dict mapping feature name → SHAP contribution
        - ``base_value``: the model's expected output (average prediction)
        - ``prediction_probability``: the actual predicted churn probability
        - ``top_positive_drivers``: top 5 features pushing toward churn
        - ``top_negative_drivers``: top 5 features pushing away from churn
    """
    df = pd.DataFrame([raw_features])
    df = build_features(df, training=False)
    df_valid = validate_inference_data(df)

    model = ModelRegistry.instance().get_model()
    prob = float(model.predict_proba(df_valid)[:, 1][0])

    explainer = _get_explainer()
    shap_values = explainer.shap_values(df_valid)

    # For binary classifiers, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        values = shap_values[1][0]  # class 1 (churn) explanations
    elif shap_values.ndim == 3:
        values = shap_values[0, :, 1]
    else:
        values = shap_values[0]

    feature_names = _feature_names or list(df_valid.columns)
    contributions = dict(zip(feature_names, [round(float(v), 6) for v in values]))

    # Sort contributions by absolute magnitude
    sorted_contribs = sorted(
        contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )

    # Separate positive (pushing toward churn) and negative (away from churn)
    positive_drivers = [
        {"feature": name, "impact": val}
        for name, val in sorted_contribs
        if val > 0
    ][:5]

    negative_drivers = [
        {"feature": name, "impact": val}
        for name, val in sorted_contribs
        if val < 0
    ][:5]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])  # class 1

    return {
        "prediction_probability": round(prob, 4),
        "base_value": round(float(base_value), 4),
        "shap_values": contributions,
        "top_positive_drivers": positive_drivers,
        "top_negative_drivers": negative_drivers,
    }


def compute_global_importance() -> dict[str, Any]:
    """
    Compute global feature importance using mean |SHAP| values.

    Returns
    -------
    dict
        Contains:
        - ``feature_importance``: sorted list of features with their
          mean absolute SHAP contributions
        - ``sample_size``: how many background samples were used
    """
    background = _load_background_data()
    explainer = _get_explainer()

    shap_values = explainer.shap_values(background)

    # Handle binary classifier output
    if isinstance(shap_values, list):
        values = np.array(shap_values[1])
    elif shap_values.ndim == 3:
        values = shap_values[:, :, 1]
    else:
        values = np.array(shap_values)

    feature_names = _feature_names or list(background.columns)
    mean_abs_shap = np.mean(np.abs(values), axis=0)

    importance = sorted(
        [
            {"feature": name, "mean_abs_shap": round(float(val), 6)}
            for name, val in zip(feature_names, mean_abs_shap)
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    return {
        "feature_importance": importance,
        "sample_size": len(background),
    }
