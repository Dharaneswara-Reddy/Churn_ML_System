"""
Thread-Safe Model Registry with Read-Write Lock.

Provides concurrent-safe model serving with support for hot-reload
(swapping the production model without downtime or dropped requests).

Concurrency Design:
  - ReadWriteLock allows MULTIPLE concurrent readers (predict requests)
    but EXCLUSIVE writer access (model reload).
  - This is critical in production: hundreds of inference threads must
    not block each other, but a model swap must be atomic.

Synchronization Primitives Used:
  - threading.Lock:      mutex for writer exclusivity
  - threading.Condition:  coordinate readers vs writer (wait/notify)
  - threading.local:      per-thread state (not used here but noted)

This pattern is equivalent to Java's ReentrantReadWriteLock or Go's sync.RWMutex.
"""

from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path
from typing import Any

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["api"])


class ReadWriteLock:
    """
    A readers-writer lock (shared-exclusive lock).

    Multiple threads can hold the read lock simultaneously, but only one
    thread can hold the write lock, and only when no readers are active.

    Implementation uses a Condition variable for efficient wait/notify
    instead of busy-spinning.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._readers_ok = threading.Condition(self._lock)
        self._readers: int = 0
        self._writer: bool = False

    def acquire_read(self) -> None:
        """Acquire the read lock. Blocks if a writer is active."""
        with self._lock:
            while self._writer:
                self._readers_ok.wait()
            self._readers += 1

    def release_read(self) -> None:
        """Release the read lock. Notifies waiting writers if last reader."""
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._readers_ok.notify_all()

    def acquire_write(self) -> None:
        """Acquire the write lock. Blocks until all readers release."""
        self._lock.acquire()
        while self._readers > 0 or self._writer:
            self._readers_ok.wait()
        self._writer = True

    def release_write(self) -> None:
        """Release the write lock. Notifies all waiting readers."""
        self._writer = False
        self._readers_ok.notify_all()
        self._lock.release()


class ModelRegistry:
    """
    Thread-safe, singleton model registry for production serving.

    Features:
    - Lazy loading: model is loaded on first access
    - Hot-reload: swap the model atomically without stopping inference
    - Version tracking: serves metadata alongside the model
    - Concurrency-safe: ReadWriteLock allows parallel reads during inference

    Usage:
        registry = ModelRegistry.instance()
        model = registry.get_model()         # thread-safe read
        registry.reload()                     # thread-safe write (hot-swap)
    """

    _instance: ModelRegistry | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._rw_lock = ReadWriteLock()
        self._model: Any = None
        self._model_version: str | None = None
        self._model_path: Path | None = None
        self._loaded_at: float | None = None

    @classmethod
    def instance(cls) -> ModelRegistry:
        """
        Thread-safe singleton access (double-checked locking pattern).

        The outer check avoids acquiring the lock on the hot path once
        the instance is initialized. The inner check prevents races
        between threads that both passed the outer check.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = ModelRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (used in tests)."""
        with cls._instance_lock:
            cls._instance = None

    def _load_model_from_disk(self) -> tuple[Any, str | None, Path]:
        """Load model and metadata from the production path."""
        model_path = Path(CONFIG["paths"]["production_model"])
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # noqa: S301

        # Load version from metadata if available
        metadata_path = model_path.parent / "metadata.json"
        version = None
        if metadata_path.exists():
            import json

            with open(metadata_path) as mf:
                meta = json.load(mf)
            version = meta.get("model_version")

        return model, version, model_path

    def get_model(self) -> Any:
        """
        Get the production model (thread-safe read).

        Lazy-loads on first call. Multiple inference threads can read
        the model concurrently without blocking each other.
        """
        # Fast path: model already loaded, just acquire read lock
        self._rw_lock.acquire_read()
        try:
            if self._model is not None:
                return self._model
        finally:
            self._rw_lock.release_read()

        # Slow path: need to load → acquire write lock
        self._rw_lock.acquire_write()
        try:
            # Double-check after acquiring write lock (another thread may
            # have loaded it while we waited)
            if self._model is not None:
                return self._model
            model, version, path = self._load_model_from_disk()
            self._model = model
            self._model_version = version
            self._model_path = path
            self._loaded_at = time.time()
            logger.info(
                "Model loaded into registry | version=%s | path=%s",
                version,
                path,
            )
            return self._model
        finally:
            self._rw_lock.release_write()

    def reload(self) -> None:
        """
        Hot-reload the production model (thread-safe write).

        Acquires exclusive write lock → all in-flight reads complete first,
        new reads block until the swap is done. The swap itself is an atomic
        pointer assignment.
        """
        self._rw_lock.acquire_write()
        try:
            old_version = self._model_version
            model, version, path = self._load_model_from_disk()
            self._model = model
            self._model_version = version
            self._model_path = path
            self._loaded_at = time.time()
            logger.info(
                "Model hot-reloaded | old_version=%s → new_version=%s",
                old_version,
                version,
            )
        finally:
            self._rw_lock.release_write()

    def get_info(self) -> dict[str, Any]:
        """Return model metadata (thread-safe read)."""
        self._rw_lock.acquire_read()
        try:
            return {
                "model_version": self._model_version,
                "model_path": str(self._model_path) if self._model_path else None,
                "loaded_at": self._loaded_at,
                "is_loaded": self._model is not None,
            }
        finally:
            self._rw_lock.release_read()
