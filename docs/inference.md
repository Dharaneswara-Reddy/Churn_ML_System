# `churn_system.inference` — Inference Engine & Model Contract

> **Location**: `src/churn_system/inference/`
> **Files**: `inference.py`, `model_contract.py`

---

## Overview

The `inference` package provides two capabilities:

1. **Offline inference** — a library function that loads the production model
   and predicts on a single feature row (used by scripts and the inference
   pipeline).
2. **Model contract management** — loads, validates, and caches the production
   model's metadata (`metadata.json`), which defines the feature schema that the
   serving layer depends on.

---

## File: `inference.py`

**Purpose**: Provides a standalone `run_inference()` function for offline or
library-based prediction (independent of the FastAPI server).

### Function: `_load_model()`

- Loads the production model from `CONFIG["paths"]["production_model"]`.
- Deserializes the pickle file.
- **Not cached** (unlike the API's `get_model()`). Each call loads fresh. For
  batch scripts that run infrequently, this is acceptable.

### Function: `run_inference(payload, *, threshold=None) → dict`

**Parameters**:
- `payload`: A dictionary of raw feature values (same shape as an API request body).
- `threshold`: Optional probability cutoff. Defaults to `CONFIG["inference"]["threshold"]`.

**Steps**:
1. Wraps the payload in a single-row DataFrame.
2. Applies `build_features()` (shared feature builder).
3. Validates against the inference schema (`validate_inference_data()`).
4. Runs `model.predict_proba()` and extracts the positive-class probability.
5. Applies the threshold to produce a binary prediction.

**Returns**:
```python
{
    "churn_probability": 0.73,
    "prediction": 1,
    "threshold": 0.5
}
```

**Used by**: `pipelines/inference_pipeline.py` and ad-hoc scripts.

---

## File: `model_contract.py`

**Purpose**: Loads and caches the production model's metadata. This metadata
defines the **contract** between the model and the serving layer — specifically,
which features the model expects, in what order, and what metrics it achieved.

### Function: `load_model_contract() → dict`

- Decorated with `@lru_cache(maxsize=1)` — loaded once, cached forever within
  the process.
- Calls `validate_model_bundle()` from `artifacts.py`, which:
  1. Checks that `model.pkl` exists.
  2. Checks that `metadata.json` exists alongside it.
  3. Validates that `feature_schema` is a non-empty list of strings.
  4. Validates `feature_count` consistency if present.
- Returns the parsed metadata dictionary.

### Function: `get_feature_schema() → list[str]`

- Convenience wrapper that returns `metadata["feature_schema"]` — the ordered
  list of feature names the model was trained on.
- **Used by**: `schema.py` (for inference validation) and `schema_generator.py`
  (for dynamic API request model generation).

### Function: `clear_model_contract_cache() → None`

- Clears the LRU cache, forcing the next call to `load_model_contract()` to
  re-read from disk.
- **Used when**: A new model is promoted to production and the API needs to
  pick up the updated contract without restarting.
