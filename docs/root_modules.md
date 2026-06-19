# Root-Level Modules — `schema.py`, `artifacts.py`, `mlflow_utils.py`

> **Location**: `src/churn_system/` (package root)
> **Files**: `schema.py`, `artifacts.py`, `mlflow_utils.py`

---

## Overview

These three modules live at the package root rather than in a sub-package because
they are used across multiple subsystems. They define shared contracts, artifact
validation, and experiment tracking integration.

---

## File: `schema.py`

**Purpose**: Defines the data contracts for training data and inference data
validation.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `TARGET_COLUMN` | `"Churn Value"` | The binary target column name |
| `ALLOWED_TARGET_VALUES` | `{0, 1}` | Valid values for the target column |
| `REQUIRED_COLUMNS` | *(set of 33 strings)* | All columns expected in raw training data |

### Function: `validate_training_data(df) → None`

- Checks that all 33 `REQUIRED_COLUMNS` are present in the DataFrame.
- Checks that the target column contains only `{0, 1}`.
- Raises `ValueError` with descriptive messages on failure.
- **Used by**: `training/steps/data_validation.py`.

### Function: `validate_inference_data(df) → DataFrame`

- Loads the feature schema from the production model contract
  (`get_feature_schema()`).
- Checks that all required model features are present.
- **Rejects** DataFrames that contain the target column (prevents accidental
  target leakage at inference time).
- Reorders columns to match the exact training-time feature order (critical for
  models that encode features positionally).
- **Used by**: `api/api.py` and `inference/inference.py`.

---

## File: `artifacts.py`

**Purpose**: Provides helper functions for locating, validating, and managing
model artifact bundles (a `model.pkl` paired with its `metadata.json`).

### Path Helpers

| Function | Returns |
|----------|---------|
| `production_model_path()` | Path to `model.pkl` |
| `production_model_dir()` | Parent directory of `model.pkl` |
| `production_metadata_path()` | Path to `metadata.json` (sibling of model) |
| `experiments_dir()` | Path to experiments directory |
| `experiment_dir(version)` | Path to a specific experiment version |
| `latest_experiment_dir()` | Path to the most recent experiment |
| `metadata_path_for_model(model_path)` | `model_path.parent / "metadata.json"` |

### Function: `load_metadata(metadata_path) → dict`

- Reads and parses a `metadata.json` file.
- Validates that the content is a JSON object (not array or scalar).

### Function: `validate_model_bundle(model_path, *, metadata_path=None, require_model=True) → dict`

Validates that a model artifact bundle is complete and well-formed:

1. **Model file exists** (if `require_model=True`).
2. **Metadata file exists** (looks for `metadata.json` next to the model).
3. **`feature_schema`** is a non-empty list of non-empty strings.
4. **`feature_count`** (if present) matches the length of `feature_schema`.
5. **`metrics`** (if present) is a dictionary.

Returns the validated metadata dictionary.

**Used by**: `inference/model_contract.py` when loading the model contract.

---

## File: `mlflow_utils.py`

**Purpose**: Configures MLflow tracking and provides retry-wrapped model/artifact
logging functions.

### Function: `configure_mlflow() → dict`

1. Reads MLflow configuration from `CONFIG["mlflow"]` and environment variables.
2. If `CHURN_MLFLOW_ENABLED` is `"0"`, `"false"`, or `"no"`: returns config with
   `enabled=False` (MLflow calls are skipped).
3. Otherwise: sets the tracking URI and experiment name, returns config with
   `enabled=True`.

### Function: `log_sklearn_model(*, pipeline, registered_model_name, artifact_path, tags) → str`

- Sets MLflow tags if provided.
- Calls `mlflow.sklearn.log_model()` to log the scikit-learn pipeline.
- Wrapped in `retry_with_backoff()` (max 3 retries, 1s base delay).
- Returns the model URI (e.g. `runs:/<run_id>/model`).
- **Used by**: `training/train.py` after selecting the winner.

### Function: `log_artifact(path) → None`

- Logs a file as an MLflow artifact.
- Skips if the file doesn't exist.
- Wrapped in `retry_with_backoff()` (max 2 retries, 0.5s base delay).
- **Used by**: `training/train.py` to log metadata and experiment reports.
