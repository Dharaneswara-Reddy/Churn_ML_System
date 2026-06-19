# Model Serving Package (`serving/`)

The `serving` package provides thread-safe, concurrent model hosting and hot-reloading primitives for low-latency production APIs.

## Concurrency and Hot-Reload Architecture

In a high-throughput production API, model serving must meet two critical requirements:
1. **High Concurrency**: Multiple incoming prediction requests (readers) must be able to access the model simultaneously without thread blocking.
2. **Zero Downtime Reloading (Hot-Swap)**: When a new champion model is promoted, it must replace the current model in memory atomically without restarting the API container or dropping active requests.

To achieve this, the package uses a custom **Readers-Writer Lock (Shared-Exclusive Lock)** pattern.

```
       +------------------------------------+
       |          Prediction API            |
       +------------------------------------+
          /            |             \
         /             |              \
   [Reader Thread]  [Reader Thread]  [Writer Thread (Reload)]
        |              |               |
        v              v               v
  +------------------------------------------+
  |             ModelRegistry                |
  |  +------------------------------------+  |
  |  |           ReadWriteLock            |  |
  |  +------------------------------------+  |
  |  - _model: sklearn.Pipeline           |  |
  |  - _model_version: "20260619_120000"  |  |
  +------------------------------------------+
```

---

## File and Component Index

### `model_registry.py`

Implements the thread-safe registry singleton and locking mechanisms.

#### `ReadWriteLock`
A shared-exclusive lock implemented using primitive `threading.Lock` and `threading.Condition` synchronization primitives.
*   **`acquire_read()`**: Increments the reader count. Blocks if a write lock is active.
*   **`release_read()`**: Decrements the reader count. If reader count drops to 0, notifies waiting writers.
*   **`acquire_write()`**: Blocks until the active reader count drops to 0. Sets the active writer flag to prevent new readers from entering.
*   **`release_write()`**: Releases writer exclusivity and notifies all waiting readers to proceed.

#### `ModelRegistry`
A thread-safe singleton managing the active model state.
*   **`instance()`**: Accesses the singleton instance using the **Double-Checked Locking** pattern to avoid lock overhead on the hot path.
*   **`get_model()`**: Thread-safe access to retrieve the loaded Scikit-Learn pipeline. Employs lazy-loading to load the pickle file from disk on the first prediction call.
*   **`reload()`**: Exclusively acquires the write lock to safely swap the internal model pointer to the new production model file without interrupting ongoing operations.
*   **`get_info()`**: Returns version metadata and loading timestamps safely.

---

## Key Performance Primitives

*   **Double-Checked Locking**:
    ```python
    if cls._instance is None:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ModelRegistry()
    ```
    This prevents race conditions during startup initialization while keeping subsequent calls lock-free.
*   **Atomic Pointer Swapping**: Replacing the model reference (`self._model = new_model`) is an atomic operation in the Python interpreter, ensuring readers always see a fully initialized model.
