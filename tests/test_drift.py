"""Unit tests for PSI drift calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_system.monitoring.drift import calculate_psi


class TestCalculatePSI:
    """Population Stability Index calculation tests."""

    def test_identical_distributions_return_near_zero(self):
        """When expected == actual, PSI ≈ 0."""
        rng = np.random.RandomState(42)
        data = pd.Series(rng.normal(0, 1, 1000))
        psi = calculate_psi(data, data.copy())
        assert psi < 0.01, f"PSI should be near-zero for identical data, got {psi}"

    def test_shifted_distribution_shows_drift(self):
        """A clear mean-shift should produce PSI > threshold (0.2)."""
        rng = np.random.RandomState(42)
        expected = pd.Series(rng.normal(0, 1, 1000))
        actual = pd.Series(rng.normal(3, 1, 1000))  # shifted by 3 std devs
        psi = calculate_psi(expected, actual)
        assert psi > 0.2, f"PSI should indicate drift, got {psi}"

    def test_psi_is_nonnegative(self):
        """PSI is always >= 0 by definition."""
        rng = np.random.RandomState(42)
        expected = pd.Series(rng.uniform(0, 1, 500))
        actual = pd.Series(rng.uniform(0.2, 0.8, 500))
        psi = calculate_psi(expected, actual)
        assert psi >= 0, f"PSI must be non-negative, got {psi}"

    def test_small_sample_still_computes(self):
        """PSI should handle small samples without error."""
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 5)
        actual = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5] * 5)
        psi = calculate_psi(expected, actual, bins=5)
        assert isinstance(psi, float)

    def test_custom_bin_count(self):
        """Different bin counts should all produce valid PSI values."""
        rng = np.random.RandomState(42)
        expected = pd.Series(rng.normal(0, 1, 500))
        actual = pd.Series(rng.normal(0, 1, 500))
        for bins in [5, 10, 20, 50]:
            psi = calculate_psi(expected, actual, bins=bins)
            assert isinstance(psi, float)
            assert psi >= 0
