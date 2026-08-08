---
type: Service
title: Model Serving
description: Thread-safe ModelRegistry with ReadWriteLock for concurrent prediction access and zero-downtime model hot-reload.
tags: [serving, concurrency, model-registry, hot-reload]
timestamp: 2026-06-30T00:00:00Z
---

The serving layer provides the runtime primitives that allow the [Prediction API](api.md) to serve predictions with high concurrency while supporting atomic model replacement without container restarts.

# Concurrency Architecture

In a high-throughput production API, model serving must meet two requirements:

1. **High concurrency** — multiple prediction request threads must access the model simultaneously without blocking each other.
2. **Zero-downtime hot-reload** — when a new champion model is [promoted](lifecycle.md), it must replace the current model in memory atomically without dropping active requests.

The system achieves this with a custom Readers-Writer Lock (Shared-Exclusive Lock) pattern.

# ReadWriteLock

A shared-exclusive lock built on `threading.Lock` and `threading.Condition`:

- **`acquire_read()`** — increments the reader count; blocks if a write lock is active.
- **`release_read()`** — decrements the reader count; notifies waiting writers when it drops to zero.
- **`acquire_write()`** — blocks until all active readers finish; prevents new readers from entering.
- **`release_write()`** — releases writer exclusivity and notifies all waiting readers.

# ModelRegistry

A thread-safe singleton managing the active model state.

- **`instance()`** — accesses the singleton using Double-Checked Locking to avoid lock overhead on the hot path.
- **`get_model()`** — thread-safe lazy-loading of the production scikit-learn pipeline from disk on the first call.
- **`reload()`** — exclusively acquires the write lock to swap the internal model pointer to the new production model file.
- **`get_info()`** — returns version metadata and loading timestamps safely.

# Double-Checked Locking

The singleton initialization uses double-checked locking:

```python
if cls._instance is None:
    with cls._instance_lock:
        if cls._instance is None:
            cls._instance = ModelRegistry()
```

This prevents race conditions during startup while keeping subsequent calls lock-free.

# Hot-Reload Trigger

When the [lifecycle orchestrator](lifecycle.md) promotes a new model, the serving layer's `reload()` method is called. The write lock ensures that all in-flight predictions complete before the model pointer is swapped. The Python interpreter's atomic reference assignment (`self._model = new_model`) ensures readers always see a fully initialized model object.

The [explainability](explainability.md) module's SHAP explainer cache is also reset after a model reload via `reset_explainer()`.
