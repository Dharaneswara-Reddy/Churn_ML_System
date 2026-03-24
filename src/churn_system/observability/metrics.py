"""
Prometheus metrics for API + pipelines.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

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

DRIFTING_FEATURES = Gauge(
    "churn_drifting_feature_count",
    "Number of drifting features detected by monitoring",
)

RETRAINING_RECOMMENDED = Gauge(
    "churn_retraining_recommended",
    "1 if retraining is recommended by monitoring, else 0",
)


def render_latest() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

