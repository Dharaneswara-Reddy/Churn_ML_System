---
type: Component
title: Feature Builder
description: Shared feature preparation module used by both training and inference to guarantee zero training-serving skew.
tags: [features, preprocessing, training-serving-skew]
timestamp: 2026-06-30T00:00:00Z
---

The feature builder is the single source of truth for feature preparation. The same `build_features()` function is called by both the [training pipeline](training.md) and the [Prediction API](api.md) / [inference engine](inference.md), ensuring there is zero training-serving skew.

Training-serving skew is one of the most common production ML bugs: it occurs when features are computed differently during training versus inference, causing the model to receive data it was never trained on.

# Dropped Columns

The following columns are removed from raw input because they are identifiers, contain no predictive signal, or would cause data leakage:

| Column | Reason |
|--------|--------|
| `CustomerID` | Identifier — not a feature |
| `Count` | Always 1 — no signal |
| `Churn Label` | Text version of target — leakage |
| `Churn Score` | Pre-computed score — leakage |
| `Churn Reason` | Post-hoc explanation — leakage |
| `CLTV` | Customer lifetime value — leakage risk |

# Target Column

The binary target column is `Churn Value` (0 = stayed, 1 = churned). It is dropped from the feature set if present to prevent target leakage.

# Transformation Steps

`build_features(df, training=False)`:

1. **Copies the DataFrame** — never mutates the original input.
2. **Coerces `Total Charges`** — converts from string to float (the raw data contains whitespace strings instead of nulls). Missing values are filled with `0.0`.
3. **Drops the target column** if present.
4. **Drops metadata columns** listed above.
5. Returns the cleaned DataFrame.

# Consumers

- [Training pipeline](training.md) — `training/steps/feature_engineering.py`
- [Prediction API](api.md) — single and batch prediction endpoints
- [Inference engine](inference.md) — offline inference
