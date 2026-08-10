<div align="center">

# ⚡ Enterprise Churn ML System

**An end-to-end, production-grade Machine Learning & MLOps platform for automated customer churn prediction, real-time inference, data drift detection, and self-healing lifecycle management.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Ruff](https://img.shields.io/badge/Code%20Style-Ruff-261230?style=for-the-badge&logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/Tests-71%20Passed-2ea44f?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

[Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [MLOps Lifecycle](#-mlops-lifecycle--drift-detection) • [Observability](#-observability--monitoring) • [Testing](#-testing--quality-assurance)

</div>

---

## 📌 Executive Summary

Predicting customer churn is not just a modeling problem — it is an operational systems challenge. Models degrade as customer behaviors shift, schema mismatches break API contracts, and unmonitored deployments fail silently.

The **Enterprise Churn ML System** bridges the gap between experimental data science and enterprise MLOps. It provides a complete, resilient microservice architecture that continuously:

1. **Ingests & Validates**: Enforces strict Pandera data quality contracts before training or inference.
2. **Trains & Ranks**: Automatically evaluates multiple model candidates (Logistic Regression, Random Forest, Gradient Boosting) using PR-AUC scoring.
3. **Serves via Async API**: Provides sub-5ms REST inference endpoints with built-in rate-limiting, authentication, and graceful shutdown.
4. **Monitors Data Drift**: Tracks Population Stability Index (PSI) per feature against baseline distributions in real-time.
5. **Triggers Self-Healing**: Automatically triggers automated retraining, candidate comparison, and safe model promotion/rollback.
6. **Ensures Reliability**: Implements a durable transactional outbox event store for asynchronous prediction logging and compliance auditing with PII redaction.

---

## 🏛️ System Architecture

The repository enforces strict separation of concerns across single-purpose domain modules:

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Ingress"]
        Client[HTTP Clients / Frontend] -->|POST /predict| Gateway[FastAPI Gateway]
        Gateway --> Auth[API Key Authentication & SlowAPI Rate Limiter]
    end

    subgraph ServingLayer ["Inference Engine & Registry"]
        Auth --> ModelRegistry[Model Registry - Thread-Safe Singleton]
        ModelRegistry --> SchemaGen[Dynamic Pydantic Schema Generator]
        ModelRegistry --> Predictor[Inference Engine - Scikit-Learn Pipeline]
    end

    subgraph EventLayer ["Reliability & Logging"]
        Predictor --> Outbox[SQLAlchemy Transactional Event Outbox]
        Outbox --> AuditLog[(Durable SQLite/Postgres Event Store)]
        Predictor --> Metrics[Prometheus Client Metrics Exporter]
    end

    subgraph MLOpsLayer ["Self-Healing MLOps Loop"]
        Metrics --> DriftEngine[PSI Drift Detection Engine]
        DriftEngine -->|Drift > Threshold| RetrainPipeline[Automated Training Pipeline]
        RetrainPipeline -->|PR-AUC & Schema Check| LifecycleManager[Model Promotion & Champion/Challenger]
        LifecycleManager -->|Hot-Reload Signal| ModelRegistry
    end
```

---

## 📦 Tech Stack & Enterprise Components

| Domain | Technology | Enterprise Capability |
| :--- | :--- | :--- |
| **Inference API** | `FastAPI`, `Uvicorn`, `SlowAPI` | Asynchronous REST serving, rate limiting, key auth, OpenAPI specs |
| **ML Framework** | `scikit-learn` | Pipeline modeling (LogisticRegression, RandomForest, GradientBoosting) |
| **Data Quality** | `Pandera`, `Pydantic v2` | Strict runtime schema enforcement for data ingest and API payloads |
| **Experiment Tracking** | `MLflow` | Versioned model artifact storage, metadata logging, stage transitions |
| **Drift Monitoring** | Custom `PSI` Engine | Population Stability Index calculation across numeric feature distributions |
| **Event Outbox** | `SQLAlchemy`, `SQLite / PostgreSQL` | Transactional outbox pattern for durable, fault-tolerant prediction event logs |
| **Observability** | `Prometheus Client` | Native metrics export (`/metrics`) for latency, throughput, and feature drift |
| **Packaging & CI** | `Docker`, `uv`, `Ruff`, `pytest` | Containerized microservices, ultra-fast dependency management, 70+ test suite |

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (recommended for 10x faster setup) or standard `pip`

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/GojoV339/Churn_ML_System.git
cd Churn_ML_System

# Create virtual environment and install dependencies
uv sync --all-extras
```

### 2. Train Initial Baseline Model

Execute the end-to-end training pipeline to ingest data, validate schemas, train candidate models, evaluate metrics, and save versioned artifacts:

```bash
.venv/bin/python -m churn_system.training.train
```

*Output summary:*
```text
2026-08-08 | INFO | Data validation passed (7,043 rows, 21 columns)
2026-08-08 | INFO | Evaluated Candidate [LogisticRegression]: PR-AUC = 0.648
2026-08-08 | INFO | Evaluated Candidate [RandomForestClassifier]: PR-AUC = 0.682
2026-08-08 | INFO | Evaluated Candidate [GradientBoostingClassifier]: PR-AUC = 0.714
2026-08-08 | INFO | Champion Model Selected: GradientBoostingClassifier (PR-AUC: 0.714)
2026-08-08 | INFO | Artifacts exported to models/experiments/churn_model_20260808_180000/
```

### 3. Promote Candidate to Production

Promote the latest trained model artifact to the production active serving directory:

```bash
.venv/bin/python -c "from churn_system.lifecycle.promote import promote_model; promote_model('churn_model_20260808_180000')"
```

### 4. Launch the Production API Server

```bash
.venv/bin/python -m uvicorn churn_system.api.api:app --host 0.0.0.0 --port 8000 --reload
```

Access interactive documentation and endpoints:
- **Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Interface**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **System Health**: [http://localhost:8000/health](http://localhost:8000/health)
- **Prometheus Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## 🔌 API Reference & Usage

### Prediction Endpoint (`POST /predict`)

#### Request Header & Body

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Country": "US",
    "State": "CA",
    "City": "Los Angeles",
    "Zip Code": "90001",
    "Lat Long": "34.0522, -118.2437",
    "Latitude": 34.0522,
    "Longitude": -118.2437,
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber Optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 85.5,
    "Total Charges": 1026.0
  }'
```

#### Response (`200 OK`)

```json
{
  "request_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
  "churn_probability": 0.7428,
  "prediction": 1,
  "threshold": 0.5,
  "latency_seconds": 0.0038
}
```

### Endpoints Overview

| Endpoint | Method | Auth | Rate Limit | Description |
| :--- | :---: | :---: | :---: | :--- |
| `/` | `GET` | — | — | System metadata & operational status |
| `/health` | `GET` | — | — | Liveness probe — the process is up |
| `/ready` | `GET` | — | — | Readiness probe — scores a real row, 503 if the model cannot serve |
| `/metrics` | `GET` | — | — | Prometheus scrape endpoint |
| `/predict` | `POST` | ✅ | 120/min | Single-customer churn inference |
| `/predict/batch` | `POST` | ✅ | 120/min | Batch inference (up to `CHURN_MAX_BATCH_SIZE`, default 100) |
| `/explain` | `POST` | ✅ | 120/min | SHAP explanation for one prediction |
| `/explain/global` | `GET` | ✅ | 120/min | Global feature importance, top `?limit=` (default 50) |
| `/feedback/{request_id}` | `POST` | ✅ | 120/min | Attach ground truth to a past prediction |
| `/monitoring/dashboard` | `GET` | ✅ | — | Consolidated monitoring reports |
| `/admin/reload-model` | `POST` | 🔐 admin | 5/min | Hot-reload the production model |
| `/subject/{subject_id}` | `DELETE` | 🔐 admin | 30/min | Erase all stored predictions for a customer (GDPR) |

Endpoints marked 🔐 require `CHURN_ADMIN_API_KEY` when it is set, so a leaked
prediction key cannot force model reloads or delete data.

---

## 🔄 MLOps Lifecycle & Drift Detection

The platform features an automated **Population Stability Index (PSI)** data drift monitoring system designed to detect feature distribution shifts before they cause silent prediction failures.

### Drift Threshold Spectrum

$$\text{PSI} = \sum \left( (P_{\text{actual}} - P_{\text{expected}}) \times \ln\left(\frac{P_{\text{actual}}}{P_{\text{expected}}}\right) \right)$$

- **$\text{PSI} < 0.10$**: **Nominal** — Baseline distribution matches inference distribution.
- **$0.10 \le \text{PSI} \le 0.20$**: **Moderate Shift** — Triggers warning alert metrics in Prometheus.
- **$\text{PSI} > 0.20$**: **Significant Drift** — Automates re-training workflow trigger.

A minimum production sample size (`monitoring.min_production_samples`, default 20)
must be met before drift is evaluated at all — PSI against a handful of rows is
large regardless of real drift, and would otherwise trigger spurious retraining.

### Safe Champion vs. Challenger Promotion

When retraining is executed:
1. **Schema Contract Validation**: The candidate must match the production feature names, types, and ordering; a mismatch refuses the promotion and leaves the incumbent serving.
2. **Performance Gating**: The challenger must improve on `model_promotion.metric` (default `pr_auc`) by at least `model_promotion.min_improvement` (default `0.0`, i.e. any strict improvement).
3. **Atomic Swap**: The bundle is staged in a sibling directory and moved into place with directory renames, so an interrupted promotion can never leave production without a model.
4. **Hot Reload**: The scheduler notifies each serving instance listed in `CHURN_RELOAD_ENDPOINTS`, which reloads the model and invalidates the feature-contract and SHAP caches in one operation.
5. **Automated Rollback**: An unhealthy model is reverted to the last *distinct* lineage version. Rollback is skipped in a cycle that just promoted, so a fresh model is not immediately undone.

### Ground Truth

Drift detection compares input distributions; it cannot tell whether a prediction
was *right*. Report observed outcomes to close that loop:

```bash
curl -X POST http://localhost:8000/feedback/<request_id> \
  -H "X-API-Key: $CHURN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"label": 1}'
```

Labelled predictions are merged into the retraining dataset, which the training
pipeline prefers over the original CSV when present.

---

## 📊 Observability & Monitoring

The system exposes native Prometheus metrics out-of-the-box:

```text
# HELP churn_api_requests_total Total API requests by path, method and status
# TYPE churn_api_requests_total counter
churn_api_requests_total{path="/predict",method="POST",status="200"} 1420

# HELP churn_api_request_latency_seconds Request latency histogram
# TYPE churn_api_request_latency_seconds histogram
churn_api_request_latency_seconds_bucket{path="/predict",le="0.005"} 1380

# HELP churn_drifting_feature_count Number of drifting features detected
# TYPE churn_drifting_feature_count gauge
churn_drifting_feature_count 1

# HELP churn_event_write_failures_total Prediction events that could not be persisted
# TYPE churn_event_write_failures_total counter
churn_event_write_failures_total 0
```

Pre-configured alert rules at [`observability/prometheus/alert_rules.yml`](observability/prometheus/alert_rules.yml) notify on:
- **`ChurnApiHighErrorRate`** — 5xx rate on `/predict` above 2% for 10m
- **`ChurnApiHighLatencyP95`** — p95 latency above 0.5s for 10m
- **`ChurnModelDriftDetected`** — 2 or more drifting features for 30m

> **Note on metric scope.** Drift and data-quality gauges are set by the lifecycle
> process, not the API. Prometheus scrapes the API only, so those series need a
> Pushgateway (or a scrape target on the scheduler) before the drift alert can fire
> in a multi-process deployment.

---

## 🐳 Containerized Deployment

The default stack is the API, the outbox worker, and Prometheus. Training, the
lifecycle scheduler, and migrations sit behind profiles so they are opt-in.

```bash
# Apply database migrations first (required on a fresh volume and after upgrades)
docker compose --profile migrate run --rm migrate

# Build and start api + worker + prometheus
docker compose up -d --build

# Inspect running container logs
docker compose logs -f api

# One-shot training job
docker compose --profile training run --rm train

# Run the drift → retrain → promote loop continuously
docker compose --profile lifecycle up -d scheduler
```

> `CHURN_API_KEY` has no default. The stack refuses to start without it, so a
> missing `.env` can never silently bring up an unauthenticated API. For local
> development set `CHURN_ALLOW_ANONYMOUS=1` instead.

### Environment Configuration Options

All settings live in `src/churn_system/config/settings.yaml` and can be overridden
via environment variables:

| Environment Variable | Default | Purpose |
| :--- | :--- | :--- |
| `CHURN_API_KEY` | *(required)* | API key clients must send as `X-API-Key` |
| `CHURN_ALLOW_ANONYMOUS` | `0` | Explicitly run with authentication disabled (local dev only) |
| `CHURN_ADMIN_API_KEY` | `""` | Separate credential for `/admin/*`; falls back to `CHURN_API_KEY` |
| `CHURN_INFERENCE_THRESHOLD` | `0.5` | Classification probability decision threshold |
| `CHURN_DISABLE_RATE_LIMIT` | `0` | Set to `1`/`true` to disable rate limiting. `0` keeps it **on** |
| `CHURN_MAX_BODY_BYTES` | `8388608` | Reject request bodies larger than this |
| `CHURN_LOG_FORMAT` | `text` | Logging mode (`json` for production) |
| `CHURN_EVENT_STORE_DATABASE_URL` | `sqlite:///./data/churn_events.db` | SQLAlchemy URL for the event store |
| `CHURN_RELOAD_ENDPOINTS` | `""` | Comma-separated `/admin/reload-model` URLs notified after promotion |
| `CHURN_SUBJECT_KEY_SALT` | *(dev default)* | Salt for the pseudonymous subject key — set a real secret in production |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | Peers permitted to set `X-Forwarded-For`. Never `*` |

---

## 🧪 Testing & Quality Assurance

The codebase maintains strict code quality standards, validated with comprehensive unit, integration, and concurrency tests.

```bash
# Execute full pytest suite (132 tests, 77% line coverage)
.venv/bin/python -m pytest tests/ -v

# With the coverage gate CI enforces
.venv/bin/python -m pytest tests/ --cov=churn_system --cov-fail-under=70

# Static linting and import formatting
.venv/bin/python -m ruff check src tests scripts alembic
```

### Test Coverage Highlights
- **API & Security**: fail-closed auth, per-endpoint authorization, body-size limits, and a uniform error envelope.
- **Inference & Contracts**: artifact validation, dynamic schema generation, and feature ordering.
- **Concurrency**: writer liveness under sustained reader load, and mutual exclusion during hot reload.
- **Monitoring & Drift**: PSI correctness including out-of-range, empty, and constant inputs, plus the minimum-sample guard on the retraining decision.
- **Lifecycle**: the schema interlock refusing a promotion, atomic swap cleanup, and the real rollback restore path.
- **Events & Outbox**: lease-based claiming under concurrent workers, retry release, and label recording.

---

## 📁 Repository Structure

```text
Churn_Ml_System/
├── src/churn_system/          # Core package library
│   ├── api/                   # FastAPI endpoints, middleware, schema generators
│   ├── config/                # YAML configuration & env variable overrides
│   ├── events/                # Transactional outbox event store & PII redaction
│   ├── explainability/        # SHAP explanation engines
│   ├── features/              # Deterministic feature builder (training & serving)
│   ├── inference/             # Model contract verification & inference logic
│   ├── lifecycle/             # Orchestrator, promotion, rollback & lineage tracking
│   ├── logging/               # Structured JSON & console loggers
│   ├── monitoring/            # PSI data drift engine & health checkers
│   ├── observability/         # Prometheus metrics collection
│   ├── training/              # Multi-step scikit-learn training pipeline
│   └── validation/            # Pandera data validation schemas
├── tests/                     # 70+ automated pytest unit & integration tests
├── data/                      # Dataset files & baseline reference distributions
├── models/                    # Versioned experiments & production active bundles
├── docs/                      # Architectural documentation & Graphviz diagrams
├── observability/             # Prometheus scrape configs & alert rules
├── docker-compose.yml         # Container orchestration manifest
├── Dockerfile                 # Production API microservice container
├── pyproject.toml             # Dependencies, ruff, and pytest configurations
└── README.md                  # System documentation
```

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for full details.

---

<div align="center">
Built with ❤️ for scalable, reliable, and observable Machine Learning Systems.
</div>
