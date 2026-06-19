# `churn_system.features` — Shared Feature Builder

> **Location**: `src/churn_system/features/`
> **Files**: `build_features.py`

---

## Overview

The `features` package contains the **single source of truth** for feature
preparation. The same `build_features()` function is called by both the training
pipeline and the inference pipeline, ensuring there is **zero training-serving
skew**.

Training-serving skew is one of the most common production ML bugs: it happens
when features are computed differently during training versus inference, causing
the model to receive data it was never trained on.

---

## File: `build_features.py`

**Purpose**: Transforms a raw input DataFrame (whether from training CSV or
inference API request) into a model-ready feature DataFrame.

### Constant: `DROP_COLUMNS`

```python
DROP_COLUMNS = [
    "CustomerID",    # Identifier — not a feature
    "Count",         # Always 1 — no signal
    "Churn Label",   # Text version of target — leakage
    "Churn Score",   # Pre-computed score — leakage
    "Churn Reason",  # Post-hoc explanation — leakage
    "CLTV",          # Customer lifetime value — leakage risk
]
```

These columns are dropped because they are either identifiers, contain no
predictive signal, or would cause **data leakage** (they encode information
about the target variable that would not be available at prediction time).

### Constant: `TARGET_COLUMN = "Churn Value"`

The binary target column (0 = stayed, 1 = churned).

### Function: `build_features(df, training=False) → DataFrame`

**Parameters**:
- `df`: Raw input DataFrame (training data or API request row).
- `training`: Boolean flag reserved for future training-only transformations.

**Steps**:
1. **Copies the DataFrame** — never mutates the original input.
2. **Coerces `Total Charges`** — converts from string to float (the raw data
   contains whitespace strings instead of nulls). Missing values are filled
   with `0.0`.
3. **Drops the target column** if present (prevents target leakage).
4. **Drops metadata columns** listed in `DROP_COLUMNS`.
5. Returns the cleaned DataFrame.

**Used by**:
- `training/steps/feature_engineering.py` — during model training
- `api/api.py` — during single and batch prediction
- `inference/inference.py` — during offline inference
