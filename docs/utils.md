# `churn_system.utils` — Cross-Cutting Utilities

> **Location**: `src/churn_system/utils/`
> **Files**: `retry.py`

---

## Overview

The `utils` package contains shared utility functions used across multiple modules.
Currently, it houses the retry-with-backoff mechanism that provides resilience
against transient failures in external dependencies.

---

## File: `retry.py`

**Purpose**: Provides a generic retry wrapper with configurable exponential
backoff for operations that may fail transiently (database writes, network calls,
file system operations).

### Function: `retry_with_backoff(fn, *, max_retries, base_delay, max_delay, retryable_exceptions, operation_name)`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | callable | *(required)* | Zero-argument function to execute |
| `max_retries` | int | `3` | Maximum retry attempts after initial failure |
| `base_delay` | float | `0.5` | Initial delay in seconds |
| `max_delay` | float | `8.0` | Maximum cap on delay between retries |
| `retryable_exceptions` | tuple | `(Exception,)` | Exception types that trigger a retry |
| `operation_name` | str | `"operation"` | Label for log messages |

**Behavior**:
1. Attempts to call `fn()`.
2. If it succeeds, returns the result immediately.
3. If it raises an exception in `retryable_exceptions`:
   - Logs a warning with the attempt number and error.
   - Waits for `base_delay × 2^(attempt-1)` seconds (capped at `max_delay`).
   - Retries.
4. If all retries are exhausted, logs an error and re-raises the last exception.
5. If the exception is **not** in `retryable_exceptions`, it is raised immediately
   without any retry.

**Backoff schedule example** (with defaults):

| Attempt | Delay |
|---------|-------|
| 1 (initial) | 0s (immediate) |
| 2 (1st retry) | 0.5s |
| 3 (2nd retry) | 1.0s |
| 4 (3rd retry) | 2.0s |

### Where It Is Used

| Module | Operation | Retryable Exceptions |
|--------|-----------|---------------------|
| `events/predictions.py` | SQLite/Postgres write | `OperationalError`, `OSError` |
| `mlflow_utils.py` | MLflow model logging | `ConnectionError`, `OSError`, `Exception` |
| `mlflow_utils.py` | MLflow artifact logging | `ConnectionError`, `OSError`, `Exception` |
