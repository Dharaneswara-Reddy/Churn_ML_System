# `churn_system.pipelines` — High-Level Pipeline Wrappers

> **Location**: `src/churn_system/pipelines/`
> **Files**: `training_pipeline.py`, `inference_pipeline.py`, `monitoring_pipeline.py`

---

## Overview

The `pipelines` package provides thin, high-level wrappers around the core
modules. Each pipeline catches exceptions, logs start/finish events, and serves
as a clean entry point for automation tools, CI/CD systems, or manual execution.

These are the modules you would call from a scheduler, a Docker entrypoint, or
a CI step — rather than calling the lower-level modules directly.

---

## File: `training_pipeline.py`

**Purpose**: Wraps `train.py:main()` with logging and error handling.

### Function: `run_training_pipeline() → None`

1. Logs `"Training Pipeline Started"`.
2. Calls `train_model()` (which is `train.py:main()`).
3. On success: logs `"Training completed successfully"`.
4. On exception: logs the full traceback and re-raises.
5. Logs `"Training Pipeline Finished"`.

**Run with**: `python -m churn_system.pipelines.training_pipeline`

---

## File: `inference_pipeline.py`

**Purpose**: Wraps `inference.py:run_inference()` with logging and error handling.

### Function: `run_inference_pipeline(payload: dict) → dict`

1. Logs `"Inference pipeline started"`.
2. Calls `run_inference(payload)`.
3. On success: returns the prediction result and logs completion.
4. On exception: logs the traceback and re-raises.

**Used by**: External scripts or services that need to call inference
programmatically (outside the FastAPI context).

---

## File: `monitoring_pipeline.py`

**Purpose**: Runs all monitoring checks in a single call.

### Function: `run_monitoring_pipeline() → None`

1. Logs `"Monitoring Pipeline Started"`.
2. Calls `evaluate_model_health()` — computes PSI and writes health report.
3. Calls `generate_prediction_report()` — computes prediction distribution stats.
4. On exception: logs the traceback and re-raises.
5. Logs `"Monitoring Pipeline Finished"`.

**Run with**: `python -m churn_system.pipelines.monitoring_pipeline`
