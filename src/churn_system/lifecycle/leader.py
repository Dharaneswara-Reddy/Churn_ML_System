"""
Leader election for the lifecycle scheduler.

Why this is not just a file lock
--------------------------------
``fcntl.flock`` elects exactly one leader **per host**. That is correct for a
single machine — the kernel releases the lock when the holder dies, so a crashed
leader never strands the lease — but it coordinates nothing across machines. Two
Kubernetes pods on different nodes, or two Compose stacks pointed at the same
database, each acquire their own local lock and both believe they lead. Both then
retrain, both rewrite the shared drift baseline at ``data/training_reference.csv``,
and both race the promotion into ``models/production/current``.

PostgreSQL session-level advisory locks give the same crash semantics across the
whole cluster: the lock lives on a connection, and the server drops it when that
connection ends for any reason — clean exit, kill -9, or network partition
detected by TCP keepalive. No lease renewal to get wrong, no expiry to tune.

Backend selection is automatic, from the event-store URL:

* PostgreSQL event store -> advisory lock, correct across hosts.
* SQLite event store     -> file lock, correct on one host.

That mapping is deliberate rather than configurable. A SQLite event store is
already single-node by construction, so a distributed lock would imply a guarantee
the rest of the deployment cannot keep.

The liveness gap, stated plainly
--------------------------------
An advisory lock is held by a connection. If that connection drops, the server
releases the lock immediately while this process is still inside the ``with``
block and still believes it leads. ``verify_leadership()`` exists for that: the
scheduler calls it before each cycle, so the window in which a partitioned
ex-leader can act is one cycle interval rather than unbounded. Callers that skip
the check keep the old, weaker guarantee.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import text

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger
from churn_system.observability.metrics import SCHEDULER_IS_LEADER

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])

# Namespace for the advisory-lock key. Advisory locks share one global 64-bit
# keyspace per database, so an unqualified small integer (1, 42, ...) would
# collide with any other application using the same database.
LOCK_NAMESPACE = "churn_system.lifecycle.scheduler"


def advisory_lock_key(namespace: str = LOCK_NAMESPACE) -> int:
    """
    Derive the advisory-lock key deterministically from a namespace string.

    ``pg_try_advisory_lock`` takes a *signed* 64-bit integer; a value above
    2**63-1 raises rather than wrapping, so the digest is folded into the signed
    range explicitly instead of relying on the driver to coerce it.
    """
    digest = hashlib.blake2b(namespace.encode("utf-8"), digest_size=8).digest()
    unsigned = int.from_bytes(digest, "big")
    return unsigned - 2**64 if unsigned >= 2**63 else unsigned


def _event_store_url() -> str:
    return str(
        CONFIG.get("event_store", {}).get("database_url", "sqlite:///./data/churn_events.db")
    )


def backend_name() -> str:
    """Which election backend this deployment will use. Exposed for diagnostics."""
    return "postgres-advisory" if _event_store_url().startswith("postgres") else "file"


def _lock_path() -> Path:
    configured = CONFIG.get("scheduler", {}).get("leader_lock_path")
    if configured:
        return Path(configured)
    return Path(CONFIG["paths"]["monitoring_dir"]) / ".scheduler.leader.lock"


def _always_leader() -> bool:
    return True


@contextmanager
def _file_lock():
    """
    Single-host election via ``fcntl.flock``.

    Per open-file-description and released by the kernel on process death, so a
    crashed leader never strands the lease the way a PID file would.
    """
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = open(path, "w")  # noqa: SIM115 - lifetime is managed by this contextmanager
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                logger.warning(
                    "Another scheduler holds the leader lock at %s — standing by.", path
                )
                yield False, _always_leader
                return
            raise

        handle.write(f"{os.getpid()}\n")
        handle.flush()
        logger.info(
            "Acquired scheduler leadership via file lock (pid=%d, lock=%s)",
            os.getpid(),
            path,
        )
        try:
            yield True, _always_leader
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


@contextmanager
def _postgres_advisory_lock():
    """
    Cluster-wide election via a PostgreSQL session-level advisory lock.

    The lock is bound to the connection, so it must be held open for the whole
    leadership term — it is deliberately *not* returned to the pool. Releasing is
    handled by closing the connection, which also covers every abnormal exit.
    """
    from churn_system.events.db import ENGINE

    key = advisory_lock_key()
    connection = ENGINE.connect()

    try:
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
            ).scalar()
        )

        if not acquired:
            logger.warning(
                "Another scheduler holds the PostgreSQL advisory lock (key=%d) — "
                "standing by.",
                key,
            )
            yield False, _always_leader
            return

        def still_leader() -> bool:
            """
            Confirm this connection still owns the lock.

            ``pg_locks`` is the authority. Re-calling ``pg_try_advisory_lock``
            would be wrong: advisory locks are re-entrant, so a session that
            already holds one always succeeds, and the call would report "yes"
            even for a session that had lost and silently re-taken it.
            """
            try:
                return bool(
                    connection.execute(
                        text(
                            "SELECT count(*) > 0 FROM pg_locks "
                            "WHERE locktype = 'advisory' "
                            "AND objid = :low AND classid = :high "
                            "AND pid = pg_backend_pid() AND granted"
                        ),
                        {"low": key & 0xFFFFFFFF, "high": (key >> 32) & 0xFFFFFFFF},
                    ).scalar()
                )
            except Exception:
                logger.exception(
                    "Could not confirm leadership; assuming leadership is lost so a "
                    "partitioned scheduler stops acting."
                )
                return False

        logger.info(
            "Acquired scheduler leadership via PostgreSQL advisory lock "
            "(pid=%d, key=%d)",
            os.getpid(),
            key,
        )
        try:
            yield True, still_leader
        finally:
            try:
                connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
                connection.commit()
            except Exception:
                # Closing the connection releases it anyway; an explicit unlock is
                # only a courtesy so the slot frees before TCP timeout.
                logger.warning("Explicit advisory unlock failed; relying on disconnect.")
    finally:
        connection.close()


@contextmanager
def elect_leader():
    """
    Contend for lifecycle leadership.

    Yields ``(is_leader, verify)``. ``verify()`` re-checks that leadership is still
    held and should be called before each unit of work; on the file backend it is
    always True, because a local flock cannot be lost while the process lives.
    """
    backend = _postgres_advisory_lock if backend_name() == "postgres-advisory" else _file_lock

    with backend() as (is_leader, verify):
        SCHEDULER_IS_LEADER.set(1 if is_leader else 0)
        try:
            yield is_leader, verify
        finally:
            SCHEDULER_IS_LEADER.set(0)


@contextmanager
def leader_lock():
    """
    Backwards-compatible wrapper yielding only the boolean.

    Kept because ``scheduler.leader_lock`` was the published entry point; new code
    should use :func:`elect_leader` so it can verify leadership between cycles.
    """
    with elect_leader() as (is_leader, _verify):
        yield is_leader


__all__ = [
    "LOCK_NAMESPACE",
    "advisory_lock_key",
    "backend_name",
    "elect_leader",
    "leader_lock",
]


# Typing aid for callers that store the verifier.
LeadershipVerifier = Callable[[], bool]
