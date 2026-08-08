---
type: Component
title: Inference Engine & Model Contract
description: Offline inference function and model contract (metadata) management defining the feature schema contract between the model and the serving layer.
tags: [inference, model-contract, metadata, prediction]
timestamp: 2026-06-30T00:00:00Z
---

The inference package provides two capabilities: standalone offline inference for batch scripts, and model contract management that defines the feature schema the [serving layer](serving.md) and [API](api.md) depend on.

# Offline Inference

`run_inference(payload, *, threshold=None)` provides a library function for prediction outside the FastAPI context:

1. Wraps the payload dictionary in a single-row DataFrame.
2. Applies [feature building](features.md) via the shared builder.
3. [Validates](validation.md) against the inference schema.
4. Runs `model.predict_proba()` and extracts the positive-class probability.
5. Applies the threshold (defaults to [configured](config.md) value at `inference.threshold`, default 0.5).

Returns:
```json
{
    "churn_probability": 0.73,
    "prediction": 1,
    "threshold": 0.5
}
```

Unlike the API's model loading (which uses `@lru_cache`), each call to offline inference loads the model fresh from disk. This is acceptable for batch scripts that run infrequently.

# Model Contract

The model contract is the metadata (`metadata.json`) that defines the agreement between a trained model and the serving layer — specifically which features the model expects, in what order, and what metrics it achieved.

## Loading and Caching

`load_model_contract()` is decorated with `@lru_cache(maxsize=1)` — loaded once and cached for the process lifetime. It calls `validate_model_bundle()` from the [artifact management](infrastructure.md) layer, which checks:

1. `model.pkl` exists.
2. `metadata.json` exists alongside it.
3. `feature_schema` is a non-empty list of strings.
4. `feature_count` is consistent with the schema length (if present).

## Feature Schema

`get_feature_schema()` returns the ordered list of feature names the model was trained on. This is consumed by:

- The [validation](validation.md) layer for inference data validation
- The [API](api.md) for dynamic request model generation

## Cache Invalidation

`clear_model_contract_cache()` clears the LRU cache, forcing the next call to re-read from disk. This is called when a new model is [promoted to production](lifecycle.md) so the API picks up the updated contract without restarting.
