---
type: Component
title: Explainability
description: SHAP-based feature explanations providing per-prediction and global feature importance for model interpretability and compliance.
tags: [explainability, shap, xai, interpretability]
timestamp: 2026-06-30T00:00:00Z
---

The explainability system provides Explainable AI (XAI) capabilities using SHAP (SHapley Additive exPlanations) to explain why the model makes specific predictions. Explanations are served through the [Prediction API](api.md).

# Why Explainability Matters

A churn prediction that says "70% chance of churn" is useful, but knowing *why* is critical for:

- **Trust** — stakeholders can verify the model uses sensible reasoning
- **Debugging** — engineers can detect when the model relies on spurious features
- **Compliance** — regulatory requirements (e.g. GDPR Article 22) may require explaining automated decisions
- **Action** — business teams need to know which features to influence to retain customers

# SHAP Explainer

The core SHAP computation engine selects the appropriate explainer based on model type:
- **TreeExplainer** — for RandomForest and GradientBoosting models
- **KernelExplainer** — for LogisticRegression models

The explainer is lazily initialized behind a `threading.Lock` using double-checked locking, preventing race conditions when multiple [API](api.md) threads request explanations concurrently. A background dataset of 100 training samples is cached in memory.

# Per-Prediction Explanations

`explain_prediction(raw_features)` returns:

| Field | Description |
|-------|-------------|
| `shap_values` | Per-feature SHAP contribution values |
| `base_value` | Expected model output (average prediction) |
| `top_positive_drivers` | Top 5 features pushing toward churn |
| `top_negative_drivers` | Top 5 features pushing away from churn |

# Global Feature Importance

`compute_global_importance()` computes global feature importance using mean |SHAP| values across the background training sample.

# API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/explain` | POST | Per-feature SHAP explanations for a single prediction |
| `/explain/global` | GET | Global feature importance rankings |

Explanation requests run in a background thread via `asyncio.to_thread` to avoid blocking the async event loop.

# Cache Management

`reset_explainer()` clears the cached explainer. This is called after a model hot-reload in the [serving layer](serving.md) so the explainer is rebuilt against the new model.

# Example Response

```json
{
  "request_id": "a1b2c3d4",
  "prediction_probability": 0.73,
  "base_value": 0.2654,
  "shap_values": {
    "Contract_Month-to-month": 0.1832,
    "Tenure Months": -0.0921,
    "Monthly Charges": 0.0754
  },
  "top_positive_drivers": [
    {"feature": "Contract_Month-to-month", "impact": 0.1832},
    {"feature": "Monthly Charges", "impact": 0.0754}
  ],
  "top_negative_drivers": [
    {"feature": "Tenure Months", "impact": -0.0921}
  ],
  "latency_seconds": 0.0234
}
```
