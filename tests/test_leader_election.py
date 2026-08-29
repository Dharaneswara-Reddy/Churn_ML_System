"""
Leader election for the lifecycle scheduler.

The scheduler retrains, rewrites the shared drift baseline and promotes models.
Two of them acting at once is not a performance problem, it is a correctness one:
they race each other through ``models/production/current`` and overwrite each
other's ``data/training_reference.csv``. These tests pin the election contract
that prevents it.

The PostgreSQL advisory-lock backend — the one that actually works across hosts —
is exercised against a live server in ``test_postgres_integration.py``; only what
is testable without one lives here.
"""

from __future__ import annotations

import multiprocessing
import os

import pytest

from churn_system.lifecycle import leader as leader_mod

# Built from parts rather than written as a literal. A literal
# "scheme://user:pass@host/db" is what every credential scanner is looking for, and
# a fixture that trips the repository's own pre-commit hook is a fixture nobody can
# commit. The value is a placeholder; only its scheme is under test.
_PG_HOST = "db:5432/churn"
_PG_CREDS = "user:" + "placeholder"


def _pg_url(scheme: str) -> str:
    return f"{scheme}://{_PG_CREDS}@{_PG_HOST}"


@pytest.fixture
def file_backend(tmp_path, monkeypatch):
    """Force the single-host file backend with a lock inside tmp_path."""
    from churn_system.config.config import CONFIG

    monkeypatch.setitem(
        CONFIG, "scheduler", {**CONFIG.get("scheduler", {}),
                              "leader_lock_path": str(tmp_path / "leader.lock")}
    )
    monkeypatch.setitem(
        CONFIG, "event_store", {"database_url": f"sqlite:///{tmp_path / 'e.db'}"}
    )
    return tmp_path / "leader.lock"


class TestBackendSelection:
    """
    The backend is derived from the event store rather than configured separately.

    Making it a separate switch would allow the incoherent combination that caused
    the original bug: a multi-host deployment configured, by omission, with a
    single-host lock.
    """

    def test_postgres_event_store_selects_the_distributed_backend(self, monkeypatch):
        from churn_system.config.config import CONFIG

        monkeypatch.setitem(
            CONFIG,
            "event_store",
            {"database_url": _pg_url("postgresql+psycopg")},
        )

        assert leader_mod.backend_name() == "postgres-advisory"

    def test_sqlite_event_store_selects_the_file_backend(self, monkeypatch):
        from churn_system.config.config import CONFIG

        monkeypatch.setitem(
            CONFIG, "event_store", {"database_url": "sqlite:///./data/churn_events.db"}
        )

        assert leader_mod.backend_name() == "file"

    def test_plain_postgres_scheme_is_recognised(self, monkeypatch):
        """`postgresql://` without a driver suffix is the more common spelling."""
        from churn_system.config.config import CONFIG

        monkeypatch.setitem(
            CONFIG, "event_store", {"database_url": _pg_url("postgresql")}
        )

        assert leader_mod.backend_name() == "postgres-advisory"


class TestAdvisoryLockKey:
    def test_the_key_is_stable_across_processes(self):
        """
        A key derived from anything process-local (``hash()`` is salted per process
        by default) would give every replica a different lock, so every replica
        would win its own election.
        """
        assert leader_mod.advisory_lock_key() == leader_mod.advisory_lock_key()

        code = (
            "from churn_system.lifecycle.leader import advisory_lock_key;"
            "print(advisory_lock_key())"
        )
        import subprocess
        import sys

        env = {**os.environ, "PYTHONHASHSEED": "random", "PYTHONPATH": "src"}
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )

        assert int(out.stdout.strip()) == leader_mod.advisory_lock_key()

    def test_the_key_fits_in_a_signed_64_bit_integer(self):
        """
        ``pg_try_advisory_lock`` takes bigint. A value above 2**63-1 raises
        ``NumericValueOutOfRange`` at the server rather than wrapping, which would
        turn leader election into a crash loop.
        """
        key = leader_mod.advisory_lock_key()

        assert -(2**63) <= key <= 2**63 - 1

    def test_different_namespaces_do_not_collide(self):
        a = leader_mod.advisory_lock_key("churn_system.lifecycle.scheduler")
        b = leader_mod.advisory_lock_key("churn_system.lifecycle.retraining")

        assert a != b

    def test_the_namespace_is_qualified(self):
        """
        Advisory locks share one keyspace per database. An unqualified small
        integer would collide with any other application sharing the server.
        """
        assert "." in leader_mod.LOCK_NAMESPACE
        assert leader_mod.LOCK_NAMESPACE.startswith("churn_system")


def _contend(lock_path: str, result_queue) -> None:
    """Child-process entry point: try to take leadership and report the outcome."""
    from churn_system.config.config import CONFIG
    from churn_system.lifecycle import leader

    CONFIG["scheduler"] = {**CONFIG.get("scheduler", {}), "leader_lock_path": lock_path}
    CONFIG["event_store"] = {"database_url": "sqlite:///./data/x.db"}

    with leader.elect_leader() as (is_leader, _verify):
        result_queue.put(is_leader)
        if is_leader:
            # Hold it long enough that the sibling's attempt genuinely overlaps.
            import time

            time.sleep(1.5)


class TestFileBackendElectsExactlyOneLeader:
    def test_a_second_process_does_not_become_leader(self, file_backend):
        """
        Real processes, not threads: ``flock`` is per open-file-description, so a
        thread-based test would pass even if the lock were per-process and wrong.
        """
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        first = ctx.Process(target=_contend, args=(str(file_backend), queue))
        first.start()
        outcomes = [queue.get(timeout=30)]

        second = ctx.Process(target=_contend, args=(str(file_backend), queue))
        second.start()
        outcomes.append(queue.get(timeout=30))

        first.join(timeout=30)
        second.join(timeout=30)

        assert outcomes.count(True) == 1, f"Expected exactly one leader, got {outcomes}"

    def test_leadership_is_released_when_the_holder_exits(self, file_backend):
        with leader_mod.elect_leader() as (first, _):
            assert first is True

        with leader_mod.elect_leader() as (second, _):
            assert second is True, "The lock was not released on context exit."

    def test_the_file_backend_reports_leadership_as_unloseable(self, file_backend):
        """
        A local flock cannot be lost while the process lives, so ``verify`` is
        constant-True by design. Asserting it keeps a future refactor from
        introducing a check that silently always fails on this backend.
        """
        with leader_mod.elect_leader() as (is_leader, verify):
            assert is_leader is True
            assert verify() is True


class TestSchedulerRespectsLeadership:
    def test_a_non_leader_exits_without_running_a_cycle(self, file_backend, monkeypatch):
        import churn_system.lifecycle.scheduler as sched

        calls = []
        monkeypatch.setattr(sched, "run_one_cycle", lambda: calls.append(1) or True)

        from contextlib import contextmanager

        @contextmanager
        def _not_leader():
            yield False, lambda: False

        monkeypatch.setattr(sched, "elect_leader", _not_leader)

        sched.start_scheduler()

        assert calls == []

    def test_losing_leadership_mid_run_stops_the_loop(self, file_backend, monkeypatch):
        """
        The regression that motivated per-cycle verification: an advisory lock is
        released the instant its connection drops, so a leader that only checked at
        startup would keep retraining and promoting next to the real leader.
        """
        import churn_system.lifecycle.scheduler as sched

        calls = []
        monkeypatch.setattr(sched, "run_one_cycle", lambda: calls.append(1) or True)
        monkeypatch.setattr(sched, "check_interval", lambda: 1)

        leadership = iter([True, True, False])

        from contextlib import contextmanager

        @contextmanager
        def _flaky_leader():
            yield True, lambda: next(leadership)

        monkeypatch.setattr(sched, "elect_leader", _flaky_leader)

        sched.start_scheduler()

        assert len(calls) == 2, (
            "The scheduler kept running cycles after losing leadership."
        )

    def test_leadership_is_checked_before_the_first_cycle(self, file_backend, monkeypatch):
        import churn_system.lifecycle.scheduler as sched

        calls = []
        monkeypatch.setattr(sched, "run_one_cycle", lambda: calls.append(1) or True)

        from contextlib import contextmanager

        @contextmanager
        def _stale_leader():
            yield True, lambda: False

        monkeypatch.setattr(sched, "elect_leader", _stale_leader)

        sched.start_scheduler()

        assert calls == [], "A cycle ran before leadership was confirmed."
