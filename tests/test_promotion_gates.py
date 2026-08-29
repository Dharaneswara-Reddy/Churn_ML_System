"""
Promotion gate tests.

The previous gate was ``improvement > 0`` with ``min_improvement: 0.0``. On this
dataset the PR-AUC bootstrap 95% CI width was ~0.165, so a +1e-9 difference — pure
noise — promoted. Worse, the candidate that most often won a resample was a
RandomForest that predicted *zero positives* on the entire test set. These tests
pin down that a model must be independently acceptable, meaningfully better, and
better beyond the noise band.
"""

from __future__ import annotations

import pytest

from churn_system.lifecycle.model_compare import evaluate_promotion_gates


def _metrics(pr_auc, recall, precision):
    return {"pr_auc": pr_auc, "recall": recall, "precision": precision}


def _interval(lower, upper=1.0):
    return {"pr_auc": {"lower": lower, "upper": upper, "point": (lower + upper) / 2}}


@pytest.fixture(autouse=True)
def gate_config(monkeypatch):
    """Explicit gate configuration so these tests do not drift with settings.yaml."""
    from churn_system.config import config as cfg

    monkeypatch.setitem(
        cfg.CONFIG,
        "model_promotion",
        {
            "metric": "pr_auc",
            "min_improvement": 0.02,
            "min_recall": 0.40,
            "min_precision": 0.30,
            "min_pr_auc": 0.45,
            "require_statistical_significance": True,
            "no_regression": {"recall": 0.05, "precision": 0.05},
        },
    )


class TestPromotionGates:
    def test_clearly_better_candidate_is_promoted(self):
        passed, reasons = evaluate_promotion_gates(
            champion_metrics=_metrics(0.50, 0.60, 0.45),
            challenger_metrics=_metrics(0.62, 0.66, 0.48),
            challenger_intervals=_interval(lower=0.55),
        )
        assert passed is True, reasons

    def test_statistically_indistinguishable_candidate_is_rejected(self):
        """
        The exact failure mode that would have shipped a noise-driven model: a
        real-looking improvement whose CI still overlaps the champion.
        """
        passed, reasons = evaluate_promotion_gates(
            champion_metrics=_metrics(0.60, 0.60, 0.45),
            challenger_metrics=_metrics(0.63, 0.61, 0.46),
            challenger_intervals=_interval(lower=0.52),  # below champion's 0.60
        )
        assert passed is False
        assert any("within measurement noise" in r for r in reasons)

    def test_marginal_improvement_below_min_improvement_is_rejected(self):
        passed, reasons = evaluate_promotion_gates(
            champion_metrics=_metrics(0.600000, 0.60, 0.45),
            challenger_metrics=_metrics(0.600001, 0.60, 0.45),
            challenger_intervals=_interval(lower=0.59),
        )
        assert passed is False
        assert any("below the required" in r for r in reasons)

    def test_worse_candidate_is_rejected(self):
        passed, reasons = evaluate_promotion_gates(
            champion_metrics=_metrics(0.60, 0.60, 0.45),
            challenger_metrics=_metrics(0.50, 0.55, 0.42),
            challenger_intervals=_interval(lower=0.45),
        )
        assert passed is False
        assert reasons

    def test_unacceptable_recall_is_rejected_even_when_metric_improves(self):
        """
        The zero-recall RandomForest case: better on the gate metric, useless in
        production because it flags nobody.
        """
        passed, reasons = evaluate_promotion_gates(
            champion_metrics=_metrics(0.50, 0.60, 0.45),
            challenger_metrics=_metrics(0.70, 0.00, 0.00),
            challenger_intervals=_interval(lower=0.65),
        )
        assert passed is False
        assert any("min_recall" in r for r in reasons)

    def test_regression_on_a_secondary_metric_is_rejected(self):
        passed, reasons = evaluate_promotion_gates(
            champion_metrics=_metrics(0.50, 0.80, 0.45),
            challenger_metrics=_metrics(0.62, 0.60, 0.46),  # recall -0.20
            challenger_intervals=_interval(lower=0.55),
        )
        assert passed is False
        assert any("regressed" in r for r in reasons)

    def test_missing_confidence_interval_refuses_rather_than_assumes(self):
        """Absence of evidence must not be treated as evidence of significance."""
        passed, reasons = evaluate_promotion_gates(
            champion_metrics=_metrics(0.50, 0.60, 0.45),
            challenger_metrics=_metrics(0.62, 0.66, 0.48),
            challenger_intervals={},
        )
        assert passed is False
        assert any("cannot establish significance" in r for r in reasons)

    def test_decision_always_explains_itself(self):
        _, reasons = evaluate_promotion_gates(
            champion_metrics=_metrics(0.50, 0.60, 0.45),
            challenger_metrics=_metrics(0.62, 0.66, 0.48),
            challenger_intervals=_interval(lower=0.55),
        )
        assert reasons, "a promotion decision must record why it was made"
