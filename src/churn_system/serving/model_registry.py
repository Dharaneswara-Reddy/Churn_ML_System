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

import json
import pickle
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from churn_system.artifacts import verify_bundle_signature
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
        self._writers_waiting: int = 0

    def acquire_read(self) -> None:
        """
        Acquire the read lock. Blocks if a writer is active or waiting.

        Deferring to *waiting* writers is what prevents starvation. When readers
        only checked for an already-active writer, a steady stream of inference
        threads could keep the reader count above zero indefinitely and a model
        reload would wait behind them — measured at over seven seconds with just
        eight concurrent readers, and unbounded in principle.
        """
        with self._lock:
            while self._writer or self._writers_waiting > 0:
                self._readers_ok.wait()
            self._readers += 1

    def release_read(self) -> None:
        """Release the read lock. Notifies waiting writers if last reader."""
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._readers_ok.notify_all()

    def acquire_write(self) -> None:
        """
        Acquire the write lock. Blocks until all active readers release.

        ``self._lock`` is deliberately held on return — it is released by
        ``release_write``, which is what makes the critical section exclusive.
        """
        self._lock.acquire()
        self._writers_waiting += 1
        try:
            while self._readers > 0 or self._writer:
                self._readers_ok.wait()
            self._writer = True
        finally:
            self._writers_waiting -= 1

    def release_write(self) -> None:
        """Release the write lock. Notifies all waiting readers and writers."""
        self._writer = False
        self._readers_ok.notify_all()
        self._lock.release()


@dataclass(frozen=True)
class ModelBundle:
    """
    An immutable snapshot of everything one prediction needs.

    Frozen on purpose: the model object and its feature schema must never be
    replaceable independently of each other. Swapping the whole bundle in a single
    assignment is what makes a hot reload atomic from a reader's point of view.
    """

    model: Any
    metadata: Mapping[str, Any]
    feature_schema: tuple[str, ...]
    threshold: float
    version: str | None
    path: Path


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# How many times to re-read when a bundle swap races the load.
_MAX_SNAPSHOT_RETRIES = 2


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
        # One reference, replaced wholesale. The previous fields were mutated
        # independently, which is what allowed a reader to observe a new model
        # alongside an old schema.
        self._bundle: ModelBundle | None = None
        self._loaded_at: float | None = None

    # --- backwards-compatible accessors -------------------------------------
    @property
    def _model(self) -> Any:
        return self._bundle.model if self._bundle else None

    @property
    def _model_version(self) -> str | None:
        return self._bundle.version if self._bundle else None

    @property
    def _model_path(self) -> Path | None:
        return self._bundle.path if self._bundle else None

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
        """Backwards-compatible shim returning (model, version, path)."""
        bundle = self._load_bundle_from_disk()
        return bundle.model, bundle.version, bundle.path

    def _load_bundle_from_disk(self, _attempt: int = 0) -> ModelBundle:
        """
        Load one internally consistent model bundle from disk.

        Two properties this must guarantee, both of which the previous
        implementation violated:

        **Integrity before deserialisation.** ``pickle.load`` is arbitrary code
        execution by design. The training and scheduler containers mount
        ``./models`` read-write while the API mounts it read-only, and promotion
        automatically triggers a hot reload — so a compromised training job could
        write a malicious pickle that the API would execute on the next reload.
        The HMAC signature is therefore verified *before* the file is opened.

        **A single consistent snapshot.** Model and metadata used to be two
        independent reads; a bundle swap landing between them paired version A's
        model with version B's metadata, which was reproduced. Metadata is read,
        then the model, then metadata is re-read and compared — a mismatch means a
        swap raced us, so we retry once against the settled bundle.
        """
        model_path = Path(CONFIG["paths"]["production_model"])
        bundle_dir = model_path.parent
        metadata_path = bundle_dir / "metadata.json"

        # Fail closed: an unverifiable bundle is never unpickled.
        verify_bundle_signature(bundle_dir)

        metadata_before = _read_metadata(metadata_path)

        with open(model_path, "rb") as f:
            model = pickle.load(f)  # noqa: S301 - integrity verified above

        metadata_after = _read_metadata(metadata_path)
        if metadata_before != metadata_after:
            if _attempt >= _MAX_SNAPSHOT_RETRIES:
                raise RuntimeError(
                    "Model bundle kept changing while being read; refusing to serve a "
                    "model that may not match its metadata."
                )
            logger.warning(
                "Bundle changed while loading (a swap raced the read); retrying."
            )
            return self._load_bundle_from_disk(_attempt + 1)

        schema = tuple(metadata_after.get("feature_schema", ()))
        threshold = metadata_after.get("operating_threshold")
        return ModelBundle(
            model=model,
            metadata=metadata_after,
            feature_schema=schema,
            threshold=(
                float(threshold)
                if threshold is not None
                else float(CONFIG["inference"]["threshold"])
            ),
            version=metadata_after.get("model_version"),
            path=model_path,
        )

    def get_bundle(self) -> ModelBundle:
        """
        Return the current bundle as one immutable object.

        Callers must take a bundle ONCE and use it for the whole inference
        operation. Reading the schema from one call and the model from another is
        how the model/schema skew arose in the first place.
        """
        self._rw_lock.acquire_read()
        try:
            if self._bundle is not None:
                return self._bundle
        finally:
            self._rw_lock.release_read()

        # Slow path: load outside the write lock, then swap under it.
        bundle = self._load_bundle_from_disk()
        self._rw_lock.acquire_write()
        try:
            if self._bundle is None:
                self._bundle = bundle
                self._loaded_at = time.time()
                logger.info(
                    "Model loaded into registry | version=%s | path=%s",
                    bundle.version,
                    bundle.path,
                )
            return self._bundle
        finally:
            self._rw_lock.release_write()

    def get_model(self) -> Any:
        """
        Get the production model (thread-safe read).

        Prefer ``get_bundle()`` in new code: taking the model here and the schema
        from somewhere else is precisely the pattern that produced model/schema
        skew. This remains for backwards compatibility.
        """
        return self.get_bundle().model


    def reload(self) -> None:
        """
        Hot-reload the production model (thread-safe write).

        The bundle is verified and built entirely *before* the write lock is taken,
        so readers block only for a single reference assignment rather than for the
        disk read and unpickle. The swap replaces model, schema, metadata and
        threshold together — they can never be observed out of step.
        """
        bundle = self._load_bundle_from_disk()

        self._rw_lock.acquire_write()
        try:
            old_version = self._bundle.version if self._bundle else None
            self._bundle = bundle
            self._loaded_at = time.time()
        finally:
            self._rw_lock.release_write()

        self._invalidate_dependent_caches()

        logger.info(
            "Model hot-reloaded | old_version=%s → new_version=%s",
            old_version,
            bundle.version,
        )


    @staticmethod
    def _invalidate_dependent_caches() -> None:
        """Drop caches keyed to the previous model."""
        from churn_system.inference.model_contract import clear_model_contract_cache

        clear_model_contract_cache()

        try:
            from churn_system.explainability.shap_explainer import reset_explainer

            reset_explainer()
        except Exception:
            logger.exception("Failed to reset SHAP explainer after reload")

    def get_info(self) -> dict[str, Any]:
        """Return model metadata (thread-safe read)."""
        self._rw_lock.acquire_read()
        try:
            bundle = self._bundle
            return {
                "model_version": bundle.version if bundle else None,
                "model_path": str(bundle.path) if bundle else None,
                "loaded_at": self._loaded_at,
                "is_loaded": bundle is not None,
            }
        finally:
            self._rw_lock.release_read()
