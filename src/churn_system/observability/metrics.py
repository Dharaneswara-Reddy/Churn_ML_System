"""
Prometheus metrics for API, pipelines, and monitoring.

Industry-Level Metrics Categories:
  1. API Performance     — request count, latency histograms, error rates
  2. Model Quality       — calibration error, prediction entropy, Gini
  3. Data Quality        — missing values, outlier ratios, quality score
  4. Drift Detection     — feature drift count, retraining recommendations
  5. Explainability      — explanation request count, computation latency
  6. Prediction Behavior — confidence distribution, class balance
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

# ── 1. API Performance Metrics ───────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "churn_api_requests_total",
    "Total API requests",
    ["path", "method", "status"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "churn_api_request_latency_seconds",
    "API request latency (seconds)",
    ["path", "method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

INFERENCE_ERRORS_TOTAL = Counter(
    "churn_inference_errors_total",
    "Total inference errors",
)

# ── 2. Model Quality Metrics ─────────────────────────────────────────────────

CALIBRATION_ERROR = Gauge(
    "churn_model_calibration_error",
    "Expected Calibration Error (ECE) — lower is better (0 = perfect)",
)

PREDICTION_ENTROPY = Gauge(
    "churn_prediction_entropy",
    "Average binary entropy of predictions (0 = decisive, 1 = uncertain)",
)

PREDICTION_CONFIDENCE_HISTOGRAM = Histogram(
    "churn_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
)

GINI_COEFFICIENT = Gauge(
    "churn_model_gini_coefficient",
    "Gini coefficient of the prediction distribution (discriminative power)",
)

# ── 3. Data Quality Metrics ──────────────────────────────────────────────────

DATA_QUALITY_SCORE = Gauge(
    "churn_data_quality_score",
    "Overall data quality score (0.0 = worst, 1.0 = perfect)",
)

MISSING_VALUE_RATIO = Gauge(
    "churn_data_missing_value_ratio",
    "Average missing value ratio across all features",
)

OUTLIER_RATIO = Gauge(
    "churn_data_outlier_ratio",
    "Average outlier ratio across numeric features (IQR method)",
)

# ── 4. Drift Detection Metrics ───────────────────────────────────────────────

DRIFTING_FEATURES = Gauge(
    "churn_drifting_feature_count",
    "Number of drifting features detected by monitoring",
)

RETRAINING_RECOMMENDED = Gauge(
    "churn_retraining_recommended",
    "1 if retraining is recommended by monitoring, else 0",
)

# ── 5. Explainability Metrics ────────────────────────────────────────────────

EXPLANATION_REQUESTS_TOTAL = Counter(
    "churn_explanation_requests_total",
    "Total explanation requests received",
)

EXPLANATION_LATENCY_SECONDS = Summary(
    "churn_explanation_latency_seconds",
    "Time taken to compute SHAP explanations",
)

# ── 6. Prediction Behavior Metrics ───────────────────────────────────────────

PREDICTED_POSITIVE_RATE = Gauge(
    "churn_predicted_positive_rate",
    "Fraction of predictions classified as positive (churn)",
)

PREDICTED_NEGATIVE_RATE = Gauge(
    "churn_predicted_negative_rate",
    "Fraction of predictions classified as negative (no churn)",
)


def render_latest() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
