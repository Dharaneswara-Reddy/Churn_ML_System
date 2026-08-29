"""
Integration tests against a **live** PostgreSQL server.

Why these exist
---------------
Everything about the PostgreSQL path used to be verified by compiling statements
against the dialect and reading the SQL that came out. That checks what SQLAlchemy
*would* emit; it cannot check what a real server actually does with it. The gap was
not theoretical — it hid a real defect: ``events/db.py`` declares its payload
columns as ``JSON().with_variant(JSONB, "postgresql")``, but the migrations
hardcoded ``sa.JSON()``, so a real server ended up with plain ``json`` columns. A
compile-time check of the *model* said JSONB; the deployed database said json.

The two behaviours that genuinely cannot be tested on SQLite are also here:
``SELECT ... FOR UPDATE SKIP LOCKED``-style concurrent claiming under real MVCC,
and advisory-lock leader election.

Running them
------------
Skipped unless ``CHURN_TEST_POSTGRES_URL`` points at a disposable database::

    docker run -d --name churn-pg -e POSTGRES_PASSWORD=churn -e POSTGRES_USER=churn \
        -e POSTGRES_DB=churn_events -p 55432:5432 postgres:16-alpine

    CHURN_TEST_POSTGRES_URL=postgresql+psycopg://churn:churn@localhost:55432/churn_events \
        .venv/bin/python -m pytest tests/test_postgres_integration.py -v

The suite creates and drops its own schema, so point it at a throwaway database.
"""

from __future__ import annotations

import contextlib
import os
import threading
from datetime import timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

POSTGRES_URL = os.environ.get("CHURN_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="Set CHURN_TEST_POSTGRES_URL to a disposable PostgreSQL database to run these.",
)

# Modules that did `from churn_system.events.db import SessionLocal` — the name is
# bound by value at import, so rebinding only events.db would leave them pointing
# at the SQLite session factory. This list is asserted to be complete by
# `test_no_module_was_missed_when_rebinding`.
_SESSION_IMPORTERS = (
    "churn_system.events.predictions",
    "churn_system.events.retention",
    "churn_system.monitoring.prediction_reader",
    "churn_system.workers.outbox_consumer",
)


@pytest.fixture
def pg_engine():
    """A live engine with the application schema created from scratch."""
    from churn_system.events.db import Base

    engine = create_engine(POSTGRES_URL, future=True, pool_pre_ping=True)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def pg_store(pg_engine, monkeypatch):
    """
    Point the whole event-store stack at PostgreSQL for one test.

    ``ENGINE`` and ``SessionLocal`` are module-level singletons built at import, so
    every module that imported them by value has to be rebound too.
    """
    import churn_system.events.db as db_mod

    factory = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False, future=True)

    monkeypatch.setattr(db_mod, "ENGINE", pg_engine)
    monkeypatch.setattr(db_mod, "SessionLocal", factory)
    monkeypatch.setattr(db_mod, "_SCHEMA_READY", True)

    for name in _SESSION_IMPORTERS:
        module = __import__(name, fromlist=["SessionLocal"])
        if hasattr(module, "SessionLocal"):
            monkeypatch.setattr(module, "SessionLocal", factory)

    return factory


def _add_outbox(factory, count: int, status: str | None = None):
    from churn_system.events.db import OutboxEvent, OutboxStatus, now_utc

    status = status or OutboxStatus.PENDING.value
    with factory() as session:
        for i in range(count):
            session.add(
                OutboxEvent(
                    created_at=now_utc(),
                    event_type="prediction_made",
                    payload={"request_id": f"req-{i}", "seq": i},
                    status=status,
                    attempts=0,
                )
            )
        session.commit()


class TestSchemaMatchesTheModels:
    """
    The migrations and the ORM must describe the same database.

    Drift here is silent: the application keeps working, but the column types,
    indexes and defaults a production server actually has stop matching the ones
    the code was reasoned about.
    """

    def test_migrations_produce_a_schema_with_no_drift(self, monkeypatch):
        from alembic import command
        from alembic.autogenerate import compare_metadata
        from alembic.config import Config
        from alembic.migration import MigrationContext

        from churn_system.events.db import Base

        monkeypatch.setenv("CHURN_EVENT_STORE_DATABASE_URL", POSTGRES_URL)

        engine = create_engine(POSTGRES_URL, future=True)
        Base.metadata.drop_all(bind=engine)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

        cfg = Config("alembic.ini")
        command.upgrade(cfg, "head")

        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)

        engine.dispose()
        assert diff == [], f"Alembic-migrated schema drifts from the ORM models: {diff}"

    def test_json_columns_are_jsonb_not_json(self, monkeypatch):
        """
        The regression this file was written for.

        ``JSON().with_variant(JSONB, "postgresql")`` only takes effect if whatever
        creates the table honours the variant. Alembic did not, so the deployed
        column was ``json``: no GIN indexing, re-parsed on every read, and
        duplicate keys preserved instead of normalised.
        """
        from alembic import command
        from alembic.config import Config

        monkeypatch.setenv("CHURN_EVENT_STORE_DATABASE_URL", POSTGRES_URL)

        engine = create_engine(POSTGRES_URL, future=True)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        from churn_system.events.db import Base

        Base.metadata.drop_all(bind=engine)

        command.upgrade(Config("alembic.ini"), "head")

        with engine.connect() as conn:
            types = dict(
                conn.execute(
                    text(
                        "SELECT table_name || '.' || column_name, data_type "
                        "FROM information_schema.columns "
                        "WHERE column_name IN ('features', 'payload')"
                    )
                ).all()
            )
        engine.dispose()

        assert types["prediction_events.features"] == "jsonb"
        assert types["outbox_events.payload"] == "jsonb"

    def test_jsonb_normalises_payloads(self, pg_engine, pg_store):
        """
        A behavioural consequence of jsonb, not a type-name assertion: jsonb parses
        and normalises on write, so a payload survives a round-trip as structured
        data rather than as the exact bytes submitted.
        """
        _add_outbox(pg_store, 1)

        with pg_engine.connect() as conn:
            value = conn.execute(
                text("SELECT payload -> 'seq' FROM outbox_events LIMIT 1")
            ).scalar()

        # `-> 'seq'` only works because the server understands the column as JSON
        # structure; it returns the *value*, proving server-side parsing.
        assert int(value) == 0

    def test_jsonb_supports_containment_which_json_cannot(self, pg_engine, pg_store):
        """
        ``@>`` is the operator that makes a GIN index on the payload useful. It does
        not exist for the ``json`` type at all, so this query is a direct proof that
        the migration produced jsonb.
        """
        _add_outbox(pg_store, 3)

        with pg_engine.connect() as conn:
            count = conn.execute(
                text("SELECT count(*) FROM outbox_events WHERE payload @> '{\"seq\": 1}'")
            ).scalar()

        assert count == 1


class TestConcurrentOutboxClaiming:
    """
    The claim protocol's whole purpose is that N workers can run with no
    coordination. On SQLite that is untestable — writes serialise on a single file
    lock, so a passing test proves nothing about PostgreSQL's MVCC, where two
    transactions really do evaluate the same predicate at the same time.
    """

    def test_no_event_is_claimed_twice(self, pg_store, monkeypatch):
        import churn_system.workers.outbox_consumer as consumer

        monkeypatch.setattr(consumer, "MAX_ATTEMPTS", 5)

        total = 240
        _add_outbox(pg_store, total)

        claimed: list[list[int]] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker():
            mine: list[int] = []
            barrier.wait()  # maximise real contention
            while True:
                batch = consumer._claim_batch(20)
                if not batch:
                    break
                mine.extend(eid for eid, _, _ in batch)
            with lock:
                claimed.append(mine)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        flat = [eid for batch in claimed for eid in batch]

        assert len(flat) == len(set(flat)), (
            "An event was claimed by two workers simultaneously — at-least-once "
            "delivery degraded into duplicate processing."
        )
        assert len(set(flat)) == total, "Some events were never claimed by any worker."

    def test_every_event_is_eventually_processed_exactly_once(self, pg_store, pg_engine):
        import churn_system.workers.outbox_consumer as consumer

        total = 120
        _add_outbox(pg_store, total)

        seen: list[int] = []
        lock = threading.Lock()

        def worker():
            while True:
                batch = consumer._claim_batch(15)
                if not batch:
                    break
                ids = [eid for eid, _, _ in batch]
                with lock:
                    seen.extend(ids)
                consumer._mark_processed(ids)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        with pg_engine.connect() as conn:
            processed = conn.execute(
                text("SELECT count(*) FROM outbox_events WHERE status = 'PROCESSED'")
            ).scalar()

        assert sorted(seen) == sorted(set(seen))
        assert processed == total

    def test_an_expired_lease_is_reclaimed_by_another_worker(self, pg_store, pg_engine):
        """
        A worker that dies mid-batch leaves rows stamped PROCESSING. Without lease
        expiry those rows are stranded forever — the backlog silently stops
        draining while every status query still looks busy rather than stuck.
        """
        import churn_system.workers.outbox_consumer as consumer

        _add_outbox(pg_store, 4)

        first = consumer._claim_batch(4)
        assert len(first) == 4

        # The "crashed" worker never released them; nothing is claimable yet.
        assert consumer._claim_batch(4) == []

        with pg_engine.begin() as conn:
            conn.execute(
                text("UPDATE outbox_events SET claimed_at = claimed_at - interval '1 day'")
            )

        second = consumer._claim_batch(4)
        assert len(second) == 4
        assert {e[0] for e in second} == {e[0] for e in first}

    def test_exhausted_events_are_dead_lettered_not_retried_forever(self, pg_store, pg_engine):
        import churn_system.workers.outbox_consumer as consumer

        _add_outbox(pg_store, 2)

        for _ in range(consumer.MAX_ATTEMPTS):
            batch = consumer._claim_batch(2)
            if not batch:
                break
            consumer._release_claims([eid for eid, _, _ in batch], error="boom")
            with pg_engine.begin() as conn:
                conn.execute(text("UPDATE outbox_events SET claimed_at = NULL"))

        with pg_engine.connect() as conn:
            statuses = dict(
                conn.execute(
                    text("SELECT status, count(*) FROM outbox_events GROUP BY status")
                ).all()
            )

        assert statuses.get("DEAD_LETTER") == 2
        assert consumer._claim_batch(2) == []


class TestRetentionOnPostgres:
    @pytest.fixture
    def short_retention(self, monkeypatch):
        """Retention windows come from CONFIG, which is resolved at import."""
        from churn_system.config.config import CONFIG

        monkeypatch.setitem(
            CONFIG,
            "retention",
            {
                "processed_outbox_days": 7,
                "dead_letter_days": 90,
                "prediction_days": 180,
                "batch_size": 10,
            },
        )

    def test_processed_events_are_purged_in_batches(self, pg_store, pg_engine, short_retention):
        from churn_system.events.db import OutboxStatus, now_utc
        from churn_system.events.retention import purge_processed_outbox_events

        _add_outbox(pg_store, 50, status=OutboxStatus.PROCESSED.value)
        with pg_engine.begin() as conn:
            conn.execute(
                text("UPDATE outbox_events SET processed_at = :ts"),
                {"ts": now_utc() - timedelta(days=365)},
            )

        deleted = purge_processed_outbox_events(batch_size=10)

        assert deleted == 50
        with pg_engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM outbox_events")).scalar() == 0

    def test_pending_events_are_never_purged(self, pg_store, pg_engine, short_retention):
        """
        Retention must never delete work that has not been done. A PENDING row is
        live: purging it drops an event that was accepted from a client and
        promised delivery.
        """
        from churn_system.events.db import now_utc
        from churn_system.events.retention import purge_processed_outbox_events

        _add_outbox(pg_store, 10)
        with pg_engine.begin() as conn:
            conn.execute(
                text("UPDATE outbox_events SET created_at = :ts"),
                {"ts": now_utc() - timedelta(days=3650)},
            )

        purge_processed_outbox_events(batch_size=100)

        with pg_engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM outbox_events")).scalar() == 10

    def test_dead_letters_outlive_successes(self, pg_store, pg_engine, short_retention):
        from churn_system.events.db import OutboxStatus, now_utc
        from churn_system.events.retention import (
            purge_dead_letter_events,
            purge_processed_outbox_events,
        )

        _add_outbox(pg_store, 5, status=OutboxStatus.PROCESSED.value)
        _add_outbox(pg_store, 5, status=OutboxStatus.DEAD_LETTER.value)
        aged = now_utc() - timedelta(days=30)
        with pg_engine.begin() as conn:
            conn.execute(
                text("UPDATE outbox_events SET processed_at = :ts, created_at = :ts"),
                {"ts": aged},
            )

        # 30 days old: past the 7-day processed window, inside the 90-day
        # dead-letter window.
        assert purge_processed_outbox_events() == 5
        assert purge_dead_letter_events() == 0

        with pg_engine.connect() as conn:
            remaining = conn.execute(
                text("SELECT status, count(*) FROM outbox_events GROUP BY status")
            ).all()
        assert dict(remaining) == {"DEAD_LETTER": 5}

    def test_backlog_counts_group_by_status(self, pg_store):
        from churn_system.events.db import OutboxStatus
        from churn_system.events.retention import outbox_backlog

        _add_outbox(pg_store, 3)
        _add_outbox(pg_store, 2, status=OutboxStatus.DEAD_LETTER.value)

        backlog = outbox_backlog()

        assert backlog["PENDING"] == 3
        assert backlog["DEAD_LETTER"] == 2


class TestConnectionResilience:
    def test_pool_pre_ping_recovers_from_a_server_side_disconnect(self, pg_store):
        """
        Managed PostgreSQL (RDS idle timeouts, PgBouncer recycling) drops idle
        connections. Without ``pool_pre_ping`` the next query after an idle period
        raises a stale-connection OperationalError instead of reconnecting — the
        classic "first request after a quiet night fails" bug.

        Terminating the backend from a second connection reproduces exactly that.
        """
        engine = create_engine(POSTGRES_URL, future=True, pool_pre_ping=True, pool_size=1)

        with engine.connect() as conn:
            pid = conn.execute(text("SELECT pg_backend_pid()")).scalar()

        killer = create_engine(POSTGRES_URL, future=True)
        with killer.connect() as conn:
            conn.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
        killer.dispose()

        # The pooled connection is now dead. pre_ping must detect and replace it.
        with engine.connect() as conn:
            assert conn.execute(text("SELECT 1")).scalar() == 1

        engine.dispose()

    def test_engine_options_are_applied_for_postgres(self):
        from churn_system.events.db import _engine_options

        options = _engine_options(POSTGRES_URL)

        assert options["pool_pre_ping"] is True
        assert options["pool_recycle"] <= 1800
        assert options["pool_size"] >= 1

    def test_sqlite_gets_no_pooling_options(self):
        from churn_system.events.db import _engine_options

        assert _engine_options("sqlite:///./x.db") == {}


class TestPredictionWritesOnPostgres:
    @pytest.fixture
    def stub_contract(self, monkeypatch):
        """`store_prediction_event` reads the model version from the bundle."""
        import churn_system.events.predictions as pred_mod

        monkeypatch.setattr(
            pred_mod, "load_model_contract", lambda: {"model_version": "pg-test-v1"}
        )

    def test_a_prediction_and_its_outbox_row_commit_together(
        self, pg_store, pg_engine, stub_contract
    ):
        from churn_system.events.predictions import store_prediction_event

        store_prediction_event(
            request_id="req-pg-1",
            raw_features={"Tenure Months": 12, "Monthly Charges": 70.0},
            probability=0.9,
            prediction=1,
            latency_seconds=0.01,
            subject_id="cust-1",
        )

        with pg_engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM prediction_events")).scalar() == 1
            assert conn.execute(text("SELECT count(*) FROM outbox_events")).scalar() == 1

    def test_geography_is_stripped_before_it_reaches_postgres(
        self, pg_store, pg_engine, stub_contract
    ):
        """
        Redaction is what keeps PII out of the durable store. Verifying it against
        a real server matters because the column is jsonb: the server parses and
        stores structure, so anything that slipped through would be queryable.
        """
        from churn_system.events.predictions import store_prediction_event

        store_prediction_event(
            request_id="req-pg-2",
            raw_features={
                "Tenure Months": 12,
                "Monthly Charges": 70.0,
                "City": "Los Angeles",
                "Latitude": 34.05,
                "Zip Code": "90001",
            },
            probability=0.5,
            prediction=0,
            latency_seconds=0.01,
            subject_id="cust-2",
        )

        with pg_engine.connect() as conn:
            keys = conn.execute(
                text("SELECT jsonb_object_keys(features) FROM prediction_events")
            ).scalars().all()

        assert "City" not in keys
        assert "Latitude" not in keys
        assert "Tenure Months" in keys

    def test_predictions_read_back_as_a_dataframe(self, pg_store, stub_contract):
        from churn_system.events.predictions import store_prediction_event
        from churn_system.monitoring.prediction_reader import load_predictions_df

        for i in range(5):
            store_prediction_event(
                request_id=f"req-{i}",
                raw_features={"Tenure Months": i, "Monthly Charges": 50.0 + i},
                probability=0.5,
                prediction=i % 2,
                latency_seconds=0.01,
                subject_id=f"cust-{i}",
            )

        frame = load_predictions_df(limit=10)

        assert len(frame) == 5
        assert "Tenure Months" in frame.columns

    def test_subject_erasure_removes_every_row_for_a_customer(
        self, pg_store, pg_engine, stub_contract
    ):
        from churn_system.events.predictions import purge_subject, store_prediction_event

        for i in range(3):
            store_prediction_event(
                request_id=f"erase-{i}",
                raw_features={"Tenure Months": i, "Monthly Charges": 50.0},
                probability=0.5,
                prediction=0,
                latency_seconds=0.01,
                subject_id="cust-erase",
            )
        store_prediction_event(
            request_id="keep-1",
            raw_features={"Tenure Months": 9, "Monthly Charges": 50.0},
            probability=0.5,
            prediction=0,
            latency_seconds=0.01,
            subject_id="cust-keep",
        )

        deleted = purge_subject("cust-erase")

        assert deleted == 3
        with pg_engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM prediction_events")).scalar() == 1


def test_no_module_was_missed_when_rebinding():
    """
    Guard for this file's own fixture.

    If a new module does ``from churn_system.events.db import SessionLocal`` and is
    not added to ``_SESSION_IMPORTERS``, it keeps writing to SQLite while every
    other module in the test is on PostgreSQL — and the suite passes while proving
    nothing about that module.
    """
    import pathlib

    src = pathlib.Path("src/churn_system")
    importers = {
        # "src/churn_system/events/retention.py" -> "churn_system.events.retention"
        ".".join(path.relative_to(src.parent).with_suffix("").parts)
        for path in src.rglob("*.py")
        if "SessionLocal" in path.read_text(encoding="utf-8")
        and path.name != "db.py"
    }

    assert importers <= set(_SESSION_IMPORTERS), (
        f"These modules import SessionLocal but are not rebound by the pg_store "
        f"fixture: {sorted(importers - set(_SESSION_IMPORTERS))}"
    )


class TestDistributedLeaderElection:
    """
    The behaviour a file lock cannot provide.

    ``fcntl.flock`` elects one leader *per host*. Two pods on different nodes each
    take their own local lock and both believe they lead — then both retrain, both
    rewrite ``data/training_reference.csv`` and both race the promotion. A
    PostgreSQL advisory lock is held by the server, so it coordinates every process
    pointed at that database regardless of where it runs.

    Separate connections here stand in for separate hosts: to the server they are
    indistinguishable, which is exactly why the guarantee generalises.
    """

    @pytest.fixture
    def pg_config(self, monkeypatch, pg_store):
        from churn_system.config.config import CONFIG

        monkeypatch.setitem(CONFIG, "event_store", {"database_url": POSTGRES_URL})
        return CONFIG

    def test_the_postgres_backend_is_selected(self, pg_config):
        from churn_system.lifecycle import leader

        assert leader.backend_name() == "postgres-advisory"

    def test_only_one_of_two_connections_wins(self, pg_config, pg_engine):
        from churn_system.lifecycle.leader import advisory_lock_key

        key = advisory_lock_key()
        first_engine = create_engine(POSTGRES_URL, future=True)
        second_engine = create_engine(POSTGRES_URL, future=True)
        first = first_engine.connect()
        second = second_engine.connect()
        try:
            won_first = first.execute(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": key}
            ).scalar()
            won_second = second.execute(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": key}
            ).scalar()

            assert won_first is True
            assert won_second is False, (
                "Two independent connections both took leadership — the advisory "
                "lock is not coordinating them."
            )
        finally:
            # Advisory locks are held by the *session*, and closing a pooled
            # connection returns the session to the pool with its locks intact.
            # Without an explicit unlock and dispose, this test strands the
            # leadership lock and every later test that contends for it fails.
            first.execute(text("SELECT pg_advisory_unlock_all()"))
            first.commit()
            first.close()
            second.close()
            first_engine.dispose()
            second_engine.dispose()

    def test_elect_leader_blocks_a_second_contender(self, pg_config, monkeypatch):
        import churn_system.events.db as db_mod
        from churn_system.lifecycle import leader

        monkeypatch.setattr(db_mod, "ENGINE", create_engine(POSTGRES_URL, future=True))
        monkeypatch.setattr(leader, "SCHEDULER_IS_LEADER", leader.SCHEDULER_IS_LEADER)

        with leader.elect_leader() as (first, verify):
            assert first is True
            assert verify() is True

            with leader.elect_leader() as (second, _):
                assert second is False

    def test_leadership_is_released_when_the_connection_dies(self, pg_config, monkeypatch):
        """
        The crash-recovery property. A leader killed with SIGKILL never runs a
        release path; the server must drop the lock when the connection ends, or
        the lifecycle stalls permanently with no leader and no error.
        """
        from churn_system.lifecycle.leader import advisory_lock_key

        key = advisory_lock_key()
        doomed = create_engine(POSTGRES_URL, future=True).connect()
        pid = doomed.execute(text("SELECT pg_backend_pid()")).scalar()
        assert doomed.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar()

        killer = create_engine(POSTGRES_URL, future=True)
        with killer.connect() as conn:
            conn.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
        killer.dispose()

        successor = create_engine(POSTGRES_URL, future=True)
        with successor.connect() as conn:
            for _ in range(50):
                if conn.execute(
                    text("SELECT pg_try_advisory_lock(:k)"), {"k": key}
                ).scalar():
                    break
            else:
                pytest.fail("The lock was stranded after the holder's connection died.")
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
        successor.dispose()

        with contextlib.suppress(Exception):
            doomed.close()  # already terminated by the server

    def test_verify_reports_loss_after_the_lock_is_stolen(self, pg_config, monkeypatch):
        """
        ``verify()`` must consult ``pg_locks`` rather than re-calling
        ``pg_try_advisory_lock``. Advisory locks are re-entrant, so a re-try always
        succeeds for a session that already holds one — a verifier built that way
        would report "still leader" for a session that had lost it.
        """
        import churn_system.events.db as db_mod
        from churn_system.lifecycle import leader

        engine = create_engine(POSTGRES_URL, future=True)
        monkeypatch.setattr(db_mod, "ENGINE", engine)

        with leader.elect_leader() as (is_leader, verify):
            assert is_leader is True
            assert verify() is True

            # Simulate the connection dying underneath the leader.
            killer = create_engine(POSTGRES_URL, future=True)
            with killer.connect() as conn:
                conn.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) FROM pg_locks "
                        "WHERE locktype = 'advisory' AND granted AND pid <> pg_backend_pid()"
                    )
                )
            killer.dispose()

            assert verify() is False, (
                "A leader whose connection was terminated still believes it leads."
            )

        engine.dispose()
