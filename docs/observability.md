# `churn_system.observability` — Prometheus Metrics

> **Location**: `src/churn_system/observability/`
> **Files**: `metrics.py`

---

## Overview

The `observability` package defines all Prometheus metrics exported by the system.
These metrics are scraped by a Prometheus server (configured in
`observability/prometheus/prometheus.yml`) and can trigger alerts defined in
`alert_rules.yml`.

---

## File: `metrics.py`

**Purpose**: Declares all Prometheus metric instruments and provides a rendering
function for the `/metrics` endpoint.

### Metrics Defined

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `churn_api_requests_total` | **Counter** | `path`, `method`, `status` | Total API requests, broken down by endpoint, HTTP method, and status code |
| `churn_api_request_latency_seconds` | **Histogram** | `path`, `method` | Request latency distribution with buckets from 5ms to 5s |
| `churn_inference_errors_total` | **Counter** | *(none)* | Total model inference errors (exceptions during `predict_proba()`) |
| `churn_drifting_feature_count` | **Gauge** | *(none)* | Number of features currently flagged as drifting by the monitoring pipeline |
| `churn_retraining_recommended` | **Gauge** | *(none)* | Binary flag: `1` if retraining is recommended, `0` otherwise |

### Histogram Buckets

The latency histogram uses these bucket boundaries (in seconds):
`0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5`

This provides fine-grained visibility into fast requests (sub-10ms) while still
capturing slow outliers.

### Function: `render_latest() → tuple[bytes, str]`

- Calls `prometheus_client.generate_latest()` to serialize all metrics.
- Returns the bytes and the correct content type header.
- **Used by**: `api/api.py` in the `GET /metrics` endpoint.

### Alert Rules (configured externally)

The Prometheus alert rules in `observability/prometheus/alert_rules.yml` trigger
alerts based on these metrics:

| Alert | Condition | Severity |
|-------|-----------|----------|
| High error rate | Error rate > 2% over 5 minutes | Critical |
| Slow responses | p95 latency > 500ms over 5 minutes | Warning |
| Feature drift | ≥ 2 drifting features | Warning |

### Where Metrics Are Updated

| Metric | Updated In |
|--------|-----------|
| `REQUESTS_TOTAL` | `api/api.py` — after every request |
| `REQUEST_LATENCY_SECONDS` | `api/api.py` — after every request |
| `INFERENCE_ERRORS_TOTAL` | `api/api.py` — on prediction failures |
| `DRIFTING_FEATURES` | `monitoring/model_health.py` — during health evaluation |
| `RETRAINING_RECOMMENDED` | `monitoring/model_health.py` — during health evaluation |
