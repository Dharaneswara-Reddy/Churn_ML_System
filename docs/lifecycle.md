# `churn_system.lifecycle` — Model Lifecycle Management

> **Location**: `src/churn_system/lifecycle/`
> **Files**: `orchestrator.py`, `promote.py`, `rollback.py`, `lineage.py`,
> `model_compare.py`, `schema_compare.py`, `scheduler.py`

![Lifecycle Orchestration Flow](images/lifecycle_flow.png)

---

## Overview

The `lifecycle` package manages the complete model lifecycle — from deciding
whether to retrain, to comparing challenger models against the champion, to
promoting winners, to rolling back if things go wrong. It implements the
closed-loop automation that makes this system self-managing.

---

## File: `orchestrator.py`

**Purpose**: The central decision engine that ties together monitoring,
retraining, comparison, promotion, and rollback into a single automated workflow.

### Function: `run_lifecycle() → None`

Executes the following decision tree:

1. **Evaluate model health** — calls `evaluate_model_health()` which computes
   PSI drift scores.
2. **Check health report** — reads `health_report.json`. If
   `retraining_recommended` is `True`:
   - Builds a retraining dataset (original data + production logs).
   - Runs the full training pipeline (`train.py:main()`).
   - Compares the challenger against the current champion.
   - If the challenger wins and schemas are compatible: **promotes** it.
   - If the challenger loses: keeps the current champion.
3. **Run rollback check** — safety net in case the current model is unhealthy
   even after the promotion attempt.
4. Logs completion.

**Used by**: `scheduler.py` (periodic loop) and can be run directly via
`python -m churn_system.lifecycle.orchestrator`.

---

## File: `promote.py`

**Purpose**: Safely copies a trained experiment model to the production serving
directory.

### Function: `schemas_match(prod_meta, new_meta) → bool`

- Compares the `feature_schema` lists from production and challenger metadata.
- Returns `True` only if both lists are identical (same features, same order).
- **Why order matters**: The model's preprocessing pipeline encodes features
  positionally — reordering would corrupt predictions silently.

### Function: `promote_model(version: str) → None`

1. Locates the experiment directory (`experiments_dir / version`).
2. Validates that `metadata.json` exists in the experiment.
3. **Schema safety check**: If a production model already exists, compares
   schemas. If they don't match, promotion is **blocked** and an error is logged.
4. Removes the old production directory and copies the experiment directory
   to `production_dir/current/`.
5. **MLflow registry update**: If the model has an `mlflow_model_uri`, transitions
   its Model Registry stage to "Production" and archives previous versions.
6. **Records lineage**: Appends a record to the lineage log with version,
   metrics, dataset, trigger reason, and parent model.

**Used by**: `orchestrator.py` when the challenger wins the comparison.

---

## File: `rollback.py`

**Purpose**: Automatically reverts the production model to the previous version
if the current model is flagged as unhealthy.

### Function: `rollback_if_needed() → None`

1. Checks if `health_report.json` exists. If not, skips.
2. Reads the report. If `retraining_recommended` is `False`, the model is
   healthy — no rollback needed.
3. Reads the lineage log. If fewer than 2 entries exist, there's no previous
   model to rollback to.
4. Identifies the second-to-last model version from the lineage.
5. Copies the previous experiment directory to the production slot,
   replacing the current model.

**Used by**: `orchestrator.py` as the final safety check in every lifecycle cycle.

---

## File: `lineage.py`

**Purpose**: Maintains an append-only JSON log of all model promotions, providing
full traceability of which model was serving when, and why.

### Record Structure

Each lineage entry contains:

| Field | Description |
|-------|-------------|
| `model_version` | Timestamp-based version identifier |
| `timestamp` | UTC ISO timestamp of promotion |
| `dataset` | Path to the training data used |
| `trigger` | Why retraining happened (e.g. `"drift_retraining"`) |
| `parents_model` | Version of the model this one replaced |
| `metrics` | Dictionary of evaluation metrics at promotion time |

### Functions

- **`load_lineage() → list`**: Reads the lineage JSON file. Returns `[]` if the
  file doesn't exist.
- **`save_lineage(data)`**: Writes the lineage list to disk with pretty formatting.
- **`record_lineage(...)`**: Appends a new record and saves.

**Used by**: `promote.py` (after successful promotion) and `rollback.py` (to
find the previous model).

---

## File: `model_compare.py`

**Purpose**: Compares the production champion model against the latest trained
challenger model to decide whether promotion is warranted.

### Function: `get_latest_experiment() → Path | None`

- Globs `experiments_dir` for directories matching `churn_model_*`.
- Returns the last one (sorted alphabetically, which matches chronological order
  due to timestamp naming).

### Function: `compare_models() → bool`

1. Finds the latest experiment. Returns `False` if none exist.
2. If no production model exists, returns `True` (auto-promote first model).
3. **Schema compatibility check**: Calls `compare_feature_schemas()`. If the
   challenger has **removed features** (a breaking change), promotion is blocked.
4. **Metric comparison**: Compares `roc_auc` between champion and challenger.
   Returns `True` only if the challenger's score is strictly higher.

**Used by**: `orchestrator.py` to decide whether to promote.

---

## File: `schema_compare.py`

**Purpose**: Computes the diff between two models' feature schemas — identifying
added and removed features.

### Function: `compare_feature_schemas(prod_meta, challenger_meta) → dict`

- Loads `feature_schema` from both metadata files as sets.
- Computes set differences:
  - `added_features` = features in challenger but not in production
  - `removed_features` = features in production but not in challenger
  - `is_identical` = both sets are empty

**Used by**: `model_compare.py` to detect breaking schema changes.

---

## File: `scheduler.py`

**Purpose**: Runs the lifecycle orchestrator on a periodic timer, simulating a
production cron-like automation loop.

### Function: `start_scheduler() → None` (infinite loop)

1. Logs the current UTC timestamp.
2. Calls `run_lifecycle()`.
3. Catches any exceptions (non-fatal — logs and continues).
4. Sleeps for `CONFIG["scheduler"]["interval_seconds"]` (default: 60).
5. Repeats forever.

**Run with**: `python -m churn_system.lifecycle.scheduler`.
