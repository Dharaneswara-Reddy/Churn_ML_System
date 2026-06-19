"""Retry utilities with exponential backoff for external service calls."""

from __future__ import annotations

import logging
import time
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn,
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    operation_name: str = "operation",
):
    """
    Execute ``fn()`` with exponential backoff on transient failures.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to execute.
    max_retries : int
        Maximum number of retry attempts after the first failure.
    base_delay : float
        Initial delay in seconds (doubles on each retry).
    max_delay : float
        Cap on the delay between retries.
    retryable_exceptions : tuple
        Exception types that trigger a retry.
    operation_name : str
        Human-readable label used in log messages.

    Returns
    -------
    The return value of ``fn()`` on success.

    Raises
    ------
    The last exception if all retries are exhausted.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_retries + 2):  # 1 initial + max_retries retries
        try:
            return fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt > max_retries:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name,
                    attempt,
                    exc,
                )
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "%s attempt %d/%d failed (%s), retrying in %.1fs",
                operation_name,
                attempt,
                max_retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)

    # Should never reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]
