# Explainability Package (`explainability/`)

The `explainability` package provides **Explainable AI (XAI)** capabilities using SHAP (SHapley Additive exPlanations) to explain why the model makes specific predictions.

## Why Explainability Matters

A churn prediction model that says "70% chance of churn" is useful — but knowing **why** it says that (e.g., "because the customer has a month-to-month contract and high monthly charges") is critical for:
- **Trust**: Stakeholders can verify the model uses sensible reasoning
- **Debugging**: Engineers can detect when the model relies on spurious features
- **Compliance**: Regulatory requirements (e.g., GDPR Article 22) may require explaining automated decisions
- **Action**: Business teams need to know which features to influence to retain customers

## Architecture

```
  Client POST /explain
         |
         v
  +-------------------+
  |    FastAPI API     |
  | (asyncio.to_thread)|
  +-------------------+
         |
         v
  +-------------------+
  |  SHAP Explainer   |  <--- Lazily initialized (thread-safe)
  |  - TreeExplainer  |       for RandomForest/GradientBoosting
  |  - KernelExplainer|       for LogisticRegression
  +-------------------+
         |
         v
  +-------------------+
  |   Background      |  <--- 100 training samples cached in memory
  |   Dataset Cache   |
  +-------------------+
```

## File Index

### `shap_explainer.py`

Core SHAP computation engine.

#### Key Functions

- **`explain_prediction(raw_features)`**: Generates a per-prediction explanation. Returns:
  - `shap_values`: Per-feature SHAP contribution values
  - `base_value`: Expected model output (average prediction)
  - `top_positive_drivers`: Top 5 features pushing toward churn
  - `top_negative_drivers`: Top 5 features pushing away from churn

- **`compute_global_importance()`**: Computes global feature importance using mean |SHAP| values across the background training sample.

- **`reset_explainer()`**: Clears the cached explainer (called after model hot-reload).

#### Thread Safety
The SHAP explainer is lazily initialized behind a `threading.Lock` using the double-checked locking pattern. This prevents race conditions when multiple API threads request explanations concurrently.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/explain` | POST | Returns per-feature SHAP explanations for a single prediction |
| `/explain/global` | GET | Returns global feature importance rankings |

## Example Response (`POST /explain`)

```json
{
  "request_id": "a1b2c3d4",
  "prediction_probability": 0.73,
  "base_value": 0.2654,
  "shap_values": {
    "Contract_Month-to-month": 0.1832,
    "Tenure Months": -0.0921,
    "Monthly Charges": 0.0754,
    "Internet Service_Fiber Optic": 0.0612
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
