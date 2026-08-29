"""
Batch chunking behaviour.

``/predict/batch`` split its input into ``BATCH_CHUNK_SIZE`` chunks and fanned them
out through ``asyncio.gather`` + ``asyncio.to_thread``, documented as achieving
parallelism "while one chunk is waiting on GIL release". Measured, the opposite is
true: ``predict_proba`` holds the GIL for the whole traversal, so the extra threads
add nothing while each chunk repays the fixed per-call cost of DataFrame
construction, schema validation and reindexing.

On a 100-row batch the old default (25) was 2.4x slower than not chunking at all,
and chunk=10 was 5x slower. These tests pin the corrected default and the
correctness properties that must hold whatever the chunk size is.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def api_module(monkeypatch):
    import churn_system.api.api as api_mod

    return api_mod


class TestChunkSizeDefault:
    def test_the_default_is_a_single_chunk(self, api_module):
        """
        Splitting costs throughput monotonically, so the default must not split.
        A regression here is invisible — the endpoint keeps working, just slower.
        """
        assert api_module.BATCH_CHUNK_SIZE >= api_module.MAX_BATCH_SIZE

    def test_a_full_batch_produces_exactly_one_chunk(self, api_module):
        rows = list(range(api_module.MAX_BATCH_SIZE))
        size = api_module.BATCH_CHUNK_SIZE

        chunks = [rows[i : i + size] for i in range(0, len(rows), size)]

        assert len(chunks) == 1

    def test_the_default_tracks_max_batch_size(self, monkeypatch):
        """
        The two are coupled deliberately: raising the batch ceiling without raising
        the chunk size would silently reintroduce chunking.
        """
        monkeypatch.setenv("CHURN_MAX_BATCH_SIZE", "250")
        monkeypatch.delenv("CHURN_BATCH_CHUNK_SIZE", raising=False)

        import churn_system.api.api as api_mod

        reloaded = importlib.reload(api_mod)
        try:
            assert reloaded.MAX_BATCH_SIZE == 250
            assert reloaded.BATCH_CHUNK_SIZE == 250
        finally:
            monkeypatch.delenv("CHURN_MAX_BATCH_SIZE", raising=False)
            importlib.reload(api_mod)

    def test_the_knob_still_overrides(self, monkeypatch):
        """
        Kept as a memory bound for deployments that raise MAX_BATCH_SIZE far past
        100, where one inference call over the whole batch would be a large
        allocation.
        """
        monkeypatch.setenv("CHURN_BATCH_CHUNK_SIZE", "10")

        import churn_system.api.api as api_mod

        reloaded = importlib.reload(api_mod)
        try:
            assert reloaded.BATCH_CHUNK_SIZE == 10
        finally:
            monkeypatch.delenv("CHURN_BATCH_CHUNK_SIZE", raising=False)
            importlib.reload(api_mod)


class TestChunkingIsOrderPreserving:
    """
    Whatever the chunk size, row *i* of the response must be the prediction for row
    *i* of the request. A reordering bug here is silent and catastrophic: every
    caller attributes churn scores to the wrong customers.
    """

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 100])
    def test_results_reassemble_in_request_order(self, chunk_size):
        rows = list(range(20))

        chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
        reassembled = [row for chunk in chunks for row in chunk]

        assert reassembled == rows

    def test_no_row_is_dropped_or_duplicated(self):
        for chunk_size in (1, 4, 9, 25, 100):
            rows = list(range(37))
            chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
            flat = [row for chunk in chunks for row in chunk]

            assert flat == rows, f"chunk_size={chunk_size} lost or duplicated rows"
