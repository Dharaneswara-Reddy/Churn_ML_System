"""Unit tests for retry_with_backoff utility."""

from __future__ import annotations

import pytest

from churn_system.utils.retry import retry_with_backoff


class TestRetryWithBackoff:
    """Tests for the retry utility."""

    def test_succeeds_on_first_attempt(self):
        """No retries needed when function succeeds immediately."""
        result = retry_with_backoff(lambda: 42, operation_name="test_ok")
        assert result == 42

    def test_retries_on_failure_then_succeeds(self):
        """Should retry and return the result when eventually succeeds."""
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = retry_with_backoff(
            flaky,
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
            operation_name="flaky_test",
        )
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_exhausting_retries(self):
        """Should raise the last exception after max retries."""
        def always_fails():
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            retry_with_backoff(
                always_fails,
                max_retries=2,
                base_delay=0.01,
                retryable_exceptions=(ValueError,),
                operation_name="always_fail",
            )

    def test_non_retryable_exception_raises_immediately(self):
        """Exceptions not in retryable_exceptions should not be retried."""
        call_count = 0

        def wrong_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            retry_with_backoff(
                wrong_error,
                max_retries=3,
                base_delay=0.01,
                retryable_exceptions=(ConnectionError,),
                operation_name="wrong_error",
            )
        assert call_count == 1  # Should NOT have retried

    def test_returns_value_on_first_success(self):
        """Return value should be passed through correctly."""
        result = retry_with_backoff(
            lambda: {"key": "value"},
            operation_name="dict_return",
        )
        assert result == {"key": "value"}
