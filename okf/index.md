# Churn ML System — Knowledge Bundle

* [Prediction API](api.md) - FastAPI HTTP service for real-time churn predictions with authentication, rate limiting, batch processing, and structured error responses
* [Model Serving](serving.md) - Thread-safe ModelRegistry with ReadWriteLock for concurrent prediction access and zero-downtime model hot-reload
* [Training Pipeline](training.md) - Step-based offline training pipeline covering data ingestion, validation, feature engineering, multi-model training, evaluation, and artifact persistence
* [Feature Builder](features.md) - Shared feature preparation module used by both training and inference to guarantee zero training-serving skew
* [Inference Engine & Model Contract](inference.md) - Offline inference function and model contract management defining the feature schema between model and serving layer
* [Configuration Management](config.md) - Two-layer configuration using YAML base values with environment variable overrides
* [Event Store](events.md) - Durable prediction logging with PII redaction and transactional outbox pattern for async event processing
* [Outbox Worker](workers.md) - Asynchronous outbox event consumer with distributed concurrency safety via SQL row-level locking
* [Model Lifecycle Management](lifecycle.md) - Automated lifecycle covering orchestration, champion-vs-challenger comparison, promotion, rollback, lineage tracking, and scheduling
* [Monitoring & Observability](monitoring.md) - Drift detection via PSI, model health evaluation, prediction statistics, Prometheus metrics, and alerting
* [Data Validation](validation.md) - Schema-driven validation using Pandera from YAML definitions plus training and inference data contracts
* [Explainability](explainability.md) - SHAP-based per-prediction and global feature importance explanations
* [Cross-Cutting Infrastructure](infrastructure.md) - Structured logging, retry with backoff, pipeline wrappers, artifact management, and MLflow integration
