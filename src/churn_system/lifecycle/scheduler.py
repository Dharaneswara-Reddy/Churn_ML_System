"""
LifeCycle Scheduler

Runs the ML lifecycle pipeline periodically, under a leader lease.

Two properties this file exists to guarantee:

**Only one scheduler acts.** ``start_scheduler`` used to be a bare ``while True``
with no coordination of any kind. Two instances — a scaled replica, an overlapping
rolling deploy, or an operator running it by hand next to the container — would
both retrain, both rewrite the shared drift baseline, and both race the promotion.
Leadership is now elected by :mod:`churn_system.lifecycle.leader`, which uses a
PostgreSQL advisory lock when the event store is PostgreSQL (correct across hosts)
and a ``flock`` when it is SQLite (correct on one host, which is all a SQLite
deployment can be). Either way a crashed leader's lock is released automatically,
so another instance takes over without manual intervention.

Leadership is re-verified before every cycle rather than only at startup. A held
advisory lock disappears the moment its connection drops, so a partitioned leader
that only checked once would keep retraining and promoting for as long as it ran.

**Failures are loud.** The previous loop caught every exception into a single log
line, so a permanently broken retrain cycle was indistinguishable from a healthy
idle one — it reported nothing, exported no metric, and kept sleeping. Every cycle
now records its outcome, consecutive failures are tracked and exported, and the
loop refuses to keep pretending after a configurable number of consecutive
failures.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from churn_system.config.config import CONFIG
from churn_system.lifecycle.leader import backend_name, elect_leader, leader_lock
from churn_system.lifecycle.orchestrator import run_lifecycle
from churn_system.logging.logger import get_logger
from churn_system.observability.metrics import (
    SCHEDULER_CONSECUTIVE_FAILURES,
    SCHEDULER_FAILURES_TOTAL,
    SCHEDULER_IS_LEADER,
    SCHEDULER_LAST_SUCCESS_TIMESTAMP,
    SCHEDULER_RUNS_TOTAL,
)

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])


def check_interval() -> int:
    """Read at call time, and refuse a value that would spin or crash the loop."""
    interval = int(CONFIG["scheduler"]["interval_seconds"])
    if interval < 1:
        raise ValueError(
            f"scheduler.interval_seconds must be >= 1, got {interval}. A value of 0 "
            "spins the retrain loop continuously; a negative value crashes "
            "time.sleep, and with restart:unless-stopped that becomes an unbounded "
            "crash-and-retrain loop."
        )
    return interval


def max_consecutive_failures() -> int:
    """0 disables the circuit breaker (keep retrying forever)."""
    return int(CONFIG.get("scheduler", {}).get("max_consecutive_failures", 5))


# ``leader_lock`` moved to lifecycle/leader.py when a second, distributed backend
# was added. Re-exported because it was the published entry point.
__all__ = [
    "check_interval",
    "leader_lock",
    "max_consecutive_failures",
    "run_one_cycle",
    "start_scheduler",
]


def run_one_cycle() -> bool:
    """
    Run a single lifecycle cycle. Returns True on success.

    Failures are recorded against a metric and re-surfaced with a full traceback
    rather than being reduced to a log line nobody is watching.
    """
    try:
        outcome = run_lifecycle()
    except Exception:
        SCHEDULER_FAILURES_TOTAL.labels(operation="run_lifecycle").inc()
        SCHEDULER_RUNS_TOTAL.labels(outcome="error").inc()
        logger.exception(
            "Lifecycle execution failed — this cycle did NOT complete. "
            "Drift may be unevaluated and no model was promoted."
        )
        return False

    SCHEDULER_RUNS_TOTAL.labels(outcome="success").inc()
    SCHEDULER_LAST_SUCCESS_TIMESTAMP.set(time.time())
    logger.info("Lifecycle cycle completed: %s", outcome)
    return True


def start_scheduler() -> None:
    """
    Continuously run lifecycle checks at fixed intervals, as the elected leader.

    A non-leader exits immediately rather than idling: under Compose or Kubernetes
    the supervisor restarts it and it contends for the lock again, which is the
    desired failover behaviour without a bespoke retry loop.
    """
    interval = check_interval()
    limit = max_consecutive_failures()

    with elect_leader() as (is_leader, still_leader):
        if not is_leader:
            logger.info("Not the leader; exiting so the supervisor can retry later.")
            return

        logger.info(
            "Lifecycle scheduler started (interval=%ds, election=%s).",
            interval,
            backend_name(),
        )
        consecutive_failures = 0

        while True:
            # Re-checked every cycle, not just at startup. A PostgreSQL advisory
            # lock vanishes the instant its connection drops, so a partitioned
            # ex-leader that never re-checked would go on retraining, rewriting the
            # shared drift baseline and promoting models alongside the new leader.
            if not still_leader():
                logger.warning(
                    "Lost lifecycle leadership; exiting before this cycle so the "
                    "current leader acts alone."
                )
                SCHEDULER_IS_LEADER.set(0)
                return

            logger.info(
                "Running lifecycle check at %s UTC",
                datetime.now(timezone.utc).isoformat(),
            )

            if run_one_cycle():
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            SCHEDULER_CONSECUTIVE_FAILURES.set(consecutive_failures)

            if limit and consecutive_failures >= limit:
                # Refusing to continue is the point: a scheduler that keeps looping
                # on a permanent failure looks alive to every liveness probe while
                # doing nothing, which is worse than exiting loudly.
                logger.error(
                    "Stopping scheduler after %d consecutive failed cycles. "
                    "The lifecycle is broken and needs operator attention.",
                    consecutive_failures,
                )
                raise RuntimeError(
                    f"Lifecycle failed {consecutive_failures} consecutive times"
                )

            logger.info("Sleeping for %d seconds...", interval)
            time.sleep(interval)


if __name__ == "__main__":
    start_scheduler()
