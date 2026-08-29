"""
Scheduler leadership and failure-visibility tests.

Two defects these pin down:

* ``start_scheduler`` was a bare ``while True`` with no coordination, so two
  instances would both retrain, both rewrite the shared drift baseline, and both
  race the promotion.
* Every exception was caught into one log line, so a permanently broken retrain
  cycle looked exactly like a healthy idle one — no metric, no alert, and the loop
  kept sleeping forever.
"""

from __future__ import annotations

import threading

import pytest

from churn_system.lifecycle import scheduler as scheduler_module


@pytest.fixture
def lock_path(tmp_path, monkeypatch):
    from churn_system.config import config as cfg

    monkeypatch.setitem(cfg.CONFIG["paths"], "monitoring_dir", str(tmp_path))
    return tmp_path / ".scheduler.leader.lock"


class TestLeaderElection:
    def test_single_instance_becomes_leader(self, lock_path):
        with scheduler_module.leader_lock() as is_leader:
            assert is_leader is True

    def test_second_concurrent_instance_is_not_leader(self, lock_path):
        """The core guarantee: only one scheduler may run the lifecycle."""
        results: list[bool] = []
        second_checked = threading.Event()
        release_first = threading.Event()

        def contender():
            # Runs while the first holder still has the lock.
            with scheduler_module.leader_lock() as is_leader:
                results.append(is_leader)
            second_checked.set()

        with scheduler_module.leader_lock() as first:
            assert first is True
            thread = threading.Thread(target=contender)
            thread.start()
            assert second_checked.wait(timeout=5), "contender never finished"
            release_first.set()

        thread.join(timeout=5)
        assert results == [False], "a second scheduler must not become leader"

    def test_leadership_is_released_for_the_next_instance(self, lock_path):
        """A finished (or crashed) leader must not strand the lease."""
        with scheduler_module.leader_lock() as first:
            assert first is True

        with scheduler_module.leader_lock() as second:
            assert second is True, "lock was not released"

    def test_leader_metric_reflects_state(self, lock_path):
        from churn_system.observability.metrics import SCHEDULER_IS_LEADER

        with scheduler_module.leader_lock():
            assert SCHEDULER_IS_LEADER._value.get() == 1

        assert SCHEDULER_IS_LEADER._value.get() == 0


class TestFailureVisibility:
    def test_failure_increments_the_failure_metric(self, monkeypatch):
        from churn_system.observability.metrics import SCHEDULER_FAILURES_TOTAL

        def boom():
            raise RuntimeError("training exploded")

        monkeypatch.setattr(scheduler_module, "run_lifecycle", boom)
        before = SCHEDULER_FAILURES_TOTAL.labels(operation="run_lifecycle")._value.get()

        assert scheduler_module.run_one_cycle() is False

        after = SCHEDULER_FAILURES_TOTAL.labels(operation="run_lifecycle")._value.get()
        assert after == before + 1

    def test_failure_preserves_the_traceback(self, monkeypatch, propagating_logger,
                                             caplog):
        """An operator must be able to see WHY, not just that something failed."""
        propagating_logger("churn_system.lifecycle.scheduler")

        def boom():
            raise RuntimeError("distinctive-failure-text")

        monkeypatch.setattr(scheduler_module, "run_lifecycle", boom)

        with caplog.at_level("ERROR"):
            scheduler_module.run_one_cycle()

        assert "distinctive-failure-text" in caplog.text
        assert "Traceback" in caplog.text

    def test_failure_is_not_reported_as_success(self, monkeypatch):
        monkeypatch.setattr(
            scheduler_module, "run_lifecycle", lambda: (_ for _ in ()).throw(ValueError())
        )
        assert scheduler_module.run_one_cycle() is False

    def test_success_records_a_timestamp(self, monkeypatch):
        from churn_system.observability.metrics import (
            SCHEDULER_LAST_SUCCESS_TIMESTAMP,
        )

        monkeypatch.setattr(scheduler_module, "run_lifecycle", lambda: {"ok": True})

        assert scheduler_module.run_one_cycle() is True
        assert SCHEDULER_LAST_SUCCESS_TIMESTAMP._value.get() > 0


class TestIntervalValidation:
    @pytest.mark.parametrize("bad", [0, -1, -3600])
    def test_non_positive_interval_is_rejected(self, monkeypatch, bad):
        """
        0 spins the retrain loop; negative crashes time.sleep, and with
        restart:unless-stopped that becomes an unbounded crash-retrain loop.
        """
        from churn_system.config import config as cfg

        monkeypatch.setitem(cfg.CONFIG["scheduler"], "interval_seconds", bad)

        with pytest.raises(ValueError):
            scheduler_module.check_interval()

    def test_valid_interval_is_accepted(self, monkeypatch):
        from churn_system.config import config as cfg

        monkeypatch.setitem(cfg.CONFIG["scheduler"], "interval_seconds", 60)
        assert scheduler_module.check_interval() == 60
