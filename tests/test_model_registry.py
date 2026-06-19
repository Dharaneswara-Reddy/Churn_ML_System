"""Tests for thread-safe ModelRegistry (ReadWriteLock concurrency)."""

from __future__ import annotations

import threading
import time

from churn_system.serving.model_registry import ModelRegistry, ReadWriteLock


class TestReadWriteLock:
    """Verify shared-exclusive lock semantics."""

    def test_concurrent_readers_do_not_block(self):
        """Multiple readers should run concurrently, not sequentially."""
        lock = ReadWriteLock()
        barrier = threading.Barrier(3)
        results = []

        def reader(reader_id):
            lock.acquire_read()
            try:
                barrier.wait(timeout=2)  # all readers should reach here together
                results.append(reader_id)
            finally:
                lock.release_read()

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 3, "All 3 readers should have completed concurrently"

    def test_writer_excludes_readers(self):
        """A writer should block new readers until it releases."""
        lock = ReadWriteLock()
        order = []

        def writer():
            lock.acquire_write()
            try:
                order.append("writer_start")
                time.sleep(0.1)
                order.append("writer_end")
            finally:
                lock.release_write()

        def reader():
            time.sleep(0.02)  # ensure writer starts first
            lock.acquire_read()
            try:
                order.append("reader")
            finally:
                lock.release_read()

        t_writer = threading.Thread(target=writer)
        t_reader = threading.Thread(target=reader)
        t_writer.start()
        t_reader.start()
        t_writer.join(timeout=5)
        t_reader.join(timeout=5)

        # Reader should have started AFTER writer completed
        assert order.index("reader") > order.index("writer_end")


class TestModelRegistrySingleton:
    """Verify double-checked locking singleton pattern."""

    def setup_method(self):
        ModelRegistry.reset()

    def teardown_method(self):
        ModelRegistry.reset()

    def test_singleton_returns_same_instance(self):
        a = ModelRegistry.instance()
        b = ModelRegistry.instance()
        assert a is b

    def test_concurrent_singleton_access(self):
        """Multiple threads requesting the singleton should all get the same object."""
        instances = []
        barrier = threading.Barrier(5)

        def get_instance():
            barrier.wait(timeout=2)
            instances.append(id(ModelRegistry.instance()))

        threads = [threading.Thread(target=get_instance) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(set(instances)) == 1, "All threads should get the same singleton"

    def test_get_info_before_load(self):
        registry = ModelRegistry.instance()
        info = registry.get_info()
        assert info["is_loaded"] is False
        assert info["model_version"] is None

    def test_reset_clears_singleton(self):
        a = ModelRegistry.instance()
        ModelRegistry.reset()
        b = ModelRegistry.instance()
        assert a is not b
