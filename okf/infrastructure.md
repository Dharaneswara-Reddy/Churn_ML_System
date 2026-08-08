---
type: Reference
title: Cross-Cutting Infrastructure
description: Shared infrastructure modules including structured logging, retry with exponential backoff, pipeline wrappers, model artifact management, and MLflow integration.
tags: [infrastructure, logging, retry, mlflow, artifacts, pipelines]
timestamp: 2026-06-30T00:00:00Z
---

This concept covers the cross-cutting support modules that other subsystems depend on: structured logging, retry logic, pipeline wrappers, artifact management, and MLflow integration.

# Structured Logging

A centralized, configurable logging system used by every module. It supports two output formats controlled by the `CHURN_LOG_FORMAT` environment variable:

| Value | Format | Use Case |
|-------|--------|----------|
| `text` (default) | `2026-06-19 10:30:00 \| INFO \| module.name \| message` | Local development |
| `json` | `{"timestamp": "...", "level": "INFO", "logger": "...", "message": "..."}` | Production containers, log aggregation (ELK, Splunk, CloudWatch) |

The JSON formatter extracts ML-specific fields if present in the log record's `extra` dict:

`model_id`, `model_version`, `request_id`, `latency_ms`, `prediction`, `confidence`, `feature_hash`, `path`, `method`

## Logger Configuration

`get_logger(name, logfile)` creates a logger with:
- `RotatingFileHandler` — 5 MB max file size, 3 backup files, writing to `logs/`
- `StreamHandler` for console output
- `propagate = False` to prevent duplicate log lines

Each subsystem writes to its own log file (configured in `settings.yaml`):

| Subsystem | Log File |
|-----------|----------|
| Training | `logs/training.log` |
| API | `logs/api.log` |
| Monitoring | `logs/monitoring.log` |
| Lifecycle | `logs/lifecycle.log` |

# Retry with Exponential Backoff

`retry_with_backoff(fn, *, max_retries, base_delay, max_delay, retryable_exceptions, operation_name)` provides resilience against transient failures:

1. Attempts to call `fn()`.
2. On success, returns the result immediately.
3. On a retryable exception: logs a warning, waits `base_delay × 2^(attempt-1)` seconds (capped at `max_delay`), and retries.
4. If all retries are exhausted, re-raises the last exception.
5. Non-retryable exceptions are raised immediately.

Default backoff schedule:

| Attempt | Delay |
|---------|-------|
| 1 (initial) | 0s |
| 2 (1st retry) | 0.5s |
| 3 (2nd retry) | 1.0s |
| 4 (3rd retry) | 2.0s |

Used by the [event store](events.md) (DB writes) and MLflow integration (model/artifact logging).

# Pipeline Wrappers

Thin, high-level wrappers around core modules that catch exceptions, log start/finish events, and serve as clean entry points for automation tools:

| Pipeline | Wraps | Entry Point |
|----------|-------|-------------|
| `run_training_pipeline()` | [Training](training.md) `main()` | `python -m churn_system.pipelines.training_pipeline` |
| `run_inference_pipeline(payload)` | [Inference](inference.md) `run_inference()` | Programmatic use |
| `run_monitoring_pipeline()` | [Monitoring](monitoring.md) health + prediction report | `python -m churn_system.pipelines.monitoring_pipeline` |

These are the modules to call from schedulers, Docker entrypoints, or CI steps — rather than calling lower-level modules directly.

# Model Artifact Management

Helper functions for locating, validating, and managing model artifact bundles (a `model.pkl` paired with its `metadata.json`):

## Path Helpers

| Function | Returns |
|----------|---------|
| `production_model_path()` | Path to `model.pkl` |
| `production_model_dir()` | Parent directory of `model.pkl` |
| `production_metadata_path()` | Path to `metadata.json` |
| `experiments_dir()` | Path to experiments directory |
| `experiment_dir(version)` | Path to a specific experiment version |
| `latest_experiment_dir()` | Path to the most recent experiment |

## Bundle Validation

`validate_model_bundle(model_path)` checks that an artifact bundle is complete:

1. `model.pkl` exists (if required).
2. `metadata.json` exists alongside it.
3. `feature_schema` is a non-empty list of non-empty strings.
4. `feature_count` is consistent with schema length (if present).
5. `metrics` is a dictionary (if present).

Used by the [inference](inference.md) model contract loader.

# MLflow Integration

`configure_mlflow()` reads MLflow [configuration](config.md) and sets up tracking. If `CHURN_MLFLOW_ENABLED` is `"0"`, `"false"`, or `"no"`, all MLflow calls are skipped.

| Function | Purpose | Retry |
|----------|---------|-------|
| `log_sklearn_model(...)` | Logs a scikit-learn pipeline to MLflow and returns the model URI | 3 retries, 1s base delay |
| `log_artifact(path)` | Logs a file as an MLflow artifact (skips if file missing) | 2 retries, 0.5s base delay |

Used by the [training pipeline](training.md) after selecting the winner model.
