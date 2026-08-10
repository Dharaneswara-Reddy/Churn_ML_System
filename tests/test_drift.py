"""Unit tests for PSI drift calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


class TestPSIDegenerateInputs:
    """Edge cases where a naive PSI silently reports 'stable'."""

    def test_empty_production_raises_rather_than_returning_nan(self):
        """
        An empty production feed is a monitoring failure, not a stable distribution.

        Returning NaN makes every ``psi > threshold`` check False, so a total
        collapse of the prediction stream would read as a healthy model.
        """
        expected = pd.Series(np.random.RandomState(0).normal(0, 1, 100))
        with pytest.raises(ValueError):
            calculate_psi(expected, pd.Series([], dtype=float))

    def test_empty_reference_raises(self):
        actual = pd.Series(np.random.RandomState(0).normal(0, 1, 100))
        with pytest.raises(ValueError):
            calculate_psi(pd.Series([], dtype=float), actual)

    def test_all_nan_production_raises(self):
        expected = pd.Series(np.random.RandomState(0).normal(0, 1, 100))
        with pytest.raises(ValueError):
            calculate_psi(expected, pd.Series([np.nan] * 50))

    def test_out_of_range_production_is_counted_not_discarded(self):
        """
        Values outside the reference range must register as drift.

        ``np.histogram`` with fixed edges drops them, and dividing by the full sample
        length then spreads the loss across every bin — badly understating drift.
        Here 30% of production is off-scale, which must be flagged.
        """
        rng = np.random.default_rng(0)
        expected = pd.Series(rng.normal(50, 10, 5000))
        actual = expected.sample(1000, random_state=1).reset_index(drop=True).copy()
        actual.iloc[:300] = 999_999.0

        psi = calculate_psi(expected, actual)

        assert psi > 0.2, f"30% out-of-range data must flag drift, got PSI={psi}"

    def test_identical_distributions_are_exactly_zero(self):
        """Smoothing must not manufacture drift between a series and itself."""
        rng = np.random.default_rng(7)
        data = pd.Series(rng.normal(0, 1, 1000))

        assert calculate_psi(data, data.copy()) == pytest.approx(0.0, abs=1e-9)

    def test_constant_column_is_stable_against_itself(self):
        constant = pd.Series([5.0] * 200)

        assert calculate_psi(constant, constant.copy()) == pytest.approx(0.0, abs=1e-9)

    def test_sparse_reference_bins_do_not_explode(self):
        """
        A skewed reference leaves near-empty bins; resampling from it is not drift.

        Flooring already-normalised proportions used to turn ~1% of production mass
        in one sparse bin into roughly half the drift threshold.
        """
        rng = np.random.default_rng(3)
        skewed = pd.Series(
            np.concatenate([rng.normal(0, 1, 4900), rng.normal(50, 1, 100)])
        )
        resampled = skewed.sample(1000, random_state=5).reset_index(drop=True)

        assert calculate_psi(skewed, resampled) < 0.1
