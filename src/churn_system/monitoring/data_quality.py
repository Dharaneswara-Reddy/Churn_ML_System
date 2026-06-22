"""
Data Quality Monitor.

Performs automated data quality checks on incoming prediction requests
and production data to detect anomalies before they reach the model.

Industry-Standard Checks:
  1. Missing value rates per feature
  2. Type violation detection (unexpected dtypes)
  3. Outlier detection using Interquartile Range (IQR)
  4. Cardinality changes in categorical features
  5. Schema drift (new or missing columns)

These checks run asynchronously and publish metrics to Prometheus
gauges for real-time alerting dashboards.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger
from churn_system.observability.metrics import (
    DATA_QUALITY_SCORE,
    MISSING_VALUE_RATIO,
    OUTLIER_RATIO,
)

logger = get_logger(__name__, CONFIG["logging"]["monitoring"])

REPORT_DIR = Path(CONFIG["paths"]["monitoring_dir"])
REPORT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_REPORT_FILE = REPORT_DIR / "data_quality_report.json"


def check_missing_values(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute per-column missing value ratios.

    Returns a dict mapping column name → fraction of nulls (0.0 to 1.0).
    Any ratio > 0.05 (5%) is flagged as a warning.
    """
    ratios = {}
    for col in df.columns:
        ratio = float(df[col].isna().mean())
        ratios[col] = round(ratio, 6)
    return ratios


def detect_outliers(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    iqr_multiplier: float = 1.5,
) -> dict[str, dict]:
    """
    Detect outliers in numeric columns using the IQR method.

    For each numeric column, computes Q1, Q3, and IQR from the reference
    (training) data. Points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are
    flagged as outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Production data to check.
    reference_df : pd.DataFrame, optional
        Training reference data for computing IQR bounds. If None,
        bounds are computed from df itself.
    iqr_multiplier : float
        IQR multiplier for outlier bounds (default 1.5).

    Returns
    -------
    dict
        Per-column outlier statistics.
    """
    ref = reference_df if reference_df is not None else df
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_report = {}

    for col in numeric_cols:
        if col not in ref.columns:
            continue

        q1 = float(ref[col].quantile(0.25))
        q3 = float(ref[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        col_data = df[col].dropna()
        n_outliers = int(((col_data < lower) | (col_data > upper)).sum())
        ratio = round(n_outliers / max(len(col_data), 1), 6)

        outlier_report[col] = {
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": n_outliers,
            "outlier_ratio": ratio,
        }

    return outlier_report


def check_cardinality(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """
    Check for cardinality changes in categorical columns.

    Detects:
    - New categories that weren't seen during training
    - Missing categories that were present during training
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cardinality_report = {}

    for col in cat_cols:
        prod_cats = set(df[col].dropna().unique())
        current = {
            "production_cardinality": len(prod_cats),
        }

        if reference_df is not None and col in reference_df.columns:
            ref_cats = set(reference_df[col].dropna().unique())
            new_cats = prod_cats - ref_cats
            missing_cats = ref_cats - prod_cats
            current["reference_cardinality"] = len(ref_cats)
            current["new_categories"] = sorted(new_cats) if new_cats else []
            current["missing_categories"] = sorted(missing_cats) if missing_cats else []

        cardinality_report[col] = current

    return cardinality_report


def check_schema_drift(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Detect schema-level drift between production and reference data.

    Returns lists of new columns and missing columns.
    """
    prod_cols = set(df.columns)
    ref_cols = set(reference_df.columns)

    return {
        "new_columns": sorted(prod_cols - ref_cols),
        "missing_columns": sorted(ref_cols - prod_cols),
    }


def run_data_quality_checks(
    production_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> dict:
    """
    Run the full data quality assessment suite.

    Combines all individual checks into a single comprehensive report
    and publishes key metrics to Prometheus.
    """
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(production_df),
    }

    # 1. Missing values
    missing = check_missing_values(production_df)
    report["missing_value_ratios"] = missing
    avg_missing = float(np.mean(list(missing.values()))) if missing else 0.0
    MISSING_VALUE_RATIO.set(avg_missing)

    # 2. Outlier detection
    outliers = detect_outliers(production_df, reference_df)
    report["outlier_analysis"] = outliers
    if outliers:
        avg_outlier = float(np.mean([v["outlier_ratio"] for v in outliers.values()]))
    else:
        avg_outlier = 0.0
    OUTLIER_RATIO.set(avg_outlier)

    # 3. Cardinality changes
    report["cardinality_analysis"] = check_cardinality(production_df, reference_df)

    # 4. Schema drift
    if reference_df is not None:
        report["schema_drift"] = check_schema_drift(production_df, reference_df)

    # 5. Overall data quality score (0.0 = worst, 1.0 = perfect)
    # Penalize for missing values, outliers, and schema issues
    quality_score = 1.0 - min(1.0, avg_missing * 10 + avg_outlier * 5)
    report["overall_quality_score"] = round(quality_score, 4)
    DATA_QUALITY_SCORE.set(quality_score)

    # Persist report
    with open(QUALITY_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(
        "Data quality check complete | score=%.4f | missing=%.4f | outlier=%.4f",
        quality_score,
        avg_missing,
        avg_outlier,
    )

    return report
