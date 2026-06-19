# `churn_system.logging` — Structured Logging

> **Location**: `src/churn_system/logging/`
> **Files**: `logger.py`

---

## Overview

The `logging` package provides a centralized, configurable logging system used by
every module. It supports two output formats — human-readable text (for
development) and structured JSON (for production log aggregation systems like
ELK, Splunk, or CloudWatch).

---

## File: `logger.py`

**Purpose**: Creates and configures per-module loggers with rotating file handlers
and console output.

### Format Toggle

The output format is controlled by the `CHURN_LOG_FORMAT` environment variable:

| Value | Output Format | Use Case |
|-------|---------------|----------|
| `text` (default) | `2026-06-19 10:30:00 \| INFO \| module.name \| message` | Local development |
| `json` | `{"timestamp": "...", "level": "INFO", "logger": "...", "message": "..."}` | Production containers, log aggregation |

### Class: `JSONFormatter`

Custom `logging.Formatter` subclass that serializes each log record as a
single-line JSON object. In addition to the standard fields (`timestamp`,
`level`, `logger`, `message`), it extracts ML-specific fields if present:

| Extra Field | Description |
|-------------|-------------|
| `model_id` | Identifier of the model being used |
| `model_version` | Version string |
| `request_id` | API request UUID |
| `latency_ms` | Inference latency |
| `prediction` | Binary prediction value |
| `confidence` | Prediction probability |
| `feature_hash` | Hash of input features |
| `path` | HTTP endpoint path |
| `method` | HTTP method |

These fields can be passed using Python's `logger.info("msg", extra={...})`
syntax.

### Function: `get_logger(name, logfile="system.log") → Logger`

**Parameters**:
- `name`: Module name (typically `__name__`).
- `logfile`: File name for the rotating log file (e.g. `"training.log"`,
  `"api.log"`, `"monitoring.log"`).

**Behavior**:
1. Creates a logger with `logging.INFO` level.
2. Checks if the logger already has handlers (prevents duplicates on module
   reload).
3. Creates a `RotatingFileHandler`:
   - **Max file size**: 5 MB
   - **Backup count**: 3 (keeps the last 3 rotated files)
   - **Location**: `logs/` directory (created automatically)
4. Creates a `StreamHandler` for console output.
5. Both handlers use either `JSONFormatter` or the text formatter depending on
   `CHURN_LOG_FORMAT`.
6. Sets `propagate = False` to prevent duplicate log lines from parent loggers.

### Log Files by Subsystem

Each subsystem writes to its own log file (configured in `settings.yaml`):

| Subsystem | Log File |
|-----------|----------|
| Training pipeline | `logs/training.log` |
| API server | `logs/api.log` |
| Monitoring | `logs/monitoring.log` |
| Lifecycle management | `logs/lifecycle.log` |
