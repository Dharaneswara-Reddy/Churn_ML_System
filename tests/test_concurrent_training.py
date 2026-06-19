"""Tests for concurrent model training (ThreadPoolExecutor parallelism)."""

from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np
import pandas as pd

from churn_system.training.steps.model_training import (
    _train_single_candidate,
    train_candidate_models,
)


class TestTrainSingleCandidate:
    """Verify that individual candidate training works in isolation."""

    def test_returns_name_and_pipeline(self):
        """Each candidate should return (name, fitted_pipeline)."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0])

        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(max_iter=100, random_state=42)
        name, pipeline = _train_single_candidate("test_lr", estimator, X, y)

        assert name == "test_lr"
        assert hasattr(pipeline, "predict_proba")

    def test_logs_thread_name(self, caplog):
        """Training should log which thread is executing."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0])

        from sklearn.linear_model import LogisticRegression

        with caplog.at_level("INFO"):
            _train_single_candidate(
                "test_lr",
                LogisticRegression(max_iter=100, random_state=42),
                X,
                y,
            )
        assert any("Training candidate" in r.message for r in caplog.records)


class TestConcurrentTraining:
    """Verify that train_candidate_models runs candidates in parallel."""

    def test_all_candidates_trained(self):
        """All registered models should be returned as fitted pipelines."""
        X = pd.DataFrame({
            "num_1": np.random.rand(50),
            "num_2": np.random.rand(50),
            "cat_1": np.random.choice(["A", "B"], 50),
        })
        y = pd.Series(np.random.choice([0, 1], 50))

        fitted = train_candidate_models(X, y)

        # Should have all 3 default candidates
        assert len(fitted) == 3
        assert "logistic_regression" in fitted
        assert "random_forest" in fitted
        assert "gradient_boosting" in fitted

        # Each should be a fitted pipeline with predict_proba
        for name, pipeline in fitted.items():
            probs = pipeline.predict_proba(X)
            assert probs.shape == (50, 2), f"{name} should produce (n, 2) probabilities"

    def test_uses_multiple_threads(self):
        """Training should execute across multiple thread names."""
        thread_names = set()
        original_train = _train_single_candidate

        def tracking_train(name, estimator, X_train, y_train):
            thread_names.add(threading.current_thread().name)
            return original_train(name, estimator, X_train, y_train)

        X = pd.DataFrame({
            "num_1": np.random.rand(50),
            "num_2": np.random.rand(50),
            "cat_1": np.random.choice(["A", "B"], 50),
        })
        y = pd.Series(np.random.choice([0, 1], 50))

        with patch(
            "churn_system.training.steps.model_training._train_single_candidate",
            side_effect=tracking_train,
        ):
            train_candidate_models(X, y)

        # With 3 candidates and max_workers >= 2, we expect multiple threads
        assert len(thread_names) >= 2, (
            f"Expected multiple threads but got: {thread_names}"
        )
