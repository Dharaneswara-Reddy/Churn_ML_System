"""
ML methodology invariants.

Each test here pins down a specific defect the audit found. They are deliberately
assertions about *methodology*, not about metric values — a metric threshold would
either be brittle or meaningless, whereas "the split must not be on a feature" is
permanently true.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn_system.features.build_features import (
    DROP_COLUMNS,
    GEOGRAPHIC_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    build_features,
)


@pytest.fixture(scope="module")
def raw_frame(raw_dataset):
    """A frame with the full raw column contract (see the ``raw_dataset`` fixture)."""
    return pd.read_csv(raw_dataset)


class TestGeographyExcluded:
    """
    Geography must never enter the pipeline.

    Not merely dropped after preprocessing — absent from the feature frame, so it
    is never fitted, never encoded, and never present in feature_schema. It was
    98.9% of the encoded matrix, it is the PII that reached a public repository,
    and it is what /explain disclosed coordinates from.
    """

    @pytest.mark.parametrize("column", GEOGRAPHIC_COLUMNS)
    def test_geographic_column_is_absent_from_features(self, raw_frame, column):
        features = build_features(raw_frame.head(50))
        assert column not in features.columns

    def test_no_geographic_column_survives_by_any_name(self, raw_frame):
        features = build_features(raw_frame.head(50))
        suspicious = [
            c
            for c in features.columns
            if any(
                token in c.lower()
                for token in ("zip", "postal", "lat", "long", "city", "state", "country")
            )
        ]
        assert suspicious == [], f"geographic columns leaked through: {suspicious}"

    def test_geography_is_declared_not_incidental(self):
        """The exclusion must be explicit, so it survives refactoring."""
        for column in GEOGRAPHIC_COLUMNS:
            assert column in DROP_COLUMNS


class TestLeakageExcluded:
    """Target and post-outcome columns must never reach the model."""

    @pytest.mark.parametrize("column", LEAKAGE_COLUMNS)
    def test_leakage_column_is_absent(self, raw_frame, column):
        features = build_features(raw_frame.head(50))
        assert column not in features.columns

    def test_churn_score_specifically_excluded(self, raw_frame):
        """
        IBM's own churn propensity score. Retaining it yields a ~1.0 ROC-AUC and a
        completely fraudulent model, so it gets its own test.
        """
        assert "Churn Score" in LEAKAGE_COLUMNS
        assert "Churn Score" not in build_features(raw_frame.head(50)).columns

    def test_target_never_appears_in_features(self, raw_frame):
        assert TARGET_COLUMN not in build_features(raw_frame.head(50)).columns


class TestSplitMethodology:
    """
    The split must not be constructed from a model feature.

    Sorting by "Tenure Months" partitioned the data on accumulated survival — a
    monotone proxy for not having churned — producing train churn 31.5% vs test
    6.6%. Every downstream metric, the threshold, and the promotion gate inherited
    that distortion.
    """

    def test_training_module_does_not_sort_by_a_feature(self):
        source = __import__(
            "churn_system.training.train", fromlist=["train"]
        ).__file__
        text = pathlib_read(source)
        assert 'sort_values("Tenure Months")' not in text
        assert "df_sorted" not in text

    def test_split_is_stratified_on_the_target(self, raw_frame):
        from sklearn.model_selection import train_test_split

        from churn_system.training.train import GLOBAL_SEED, TEST_SIZE

        train_df, test_df = train_test_split(
            raw_frame,
            test_size=TEST_SIZE,
            stratify=raw_frame[TARGET_COLUMN],
            random_state=GLOBAL_SEED,
            shuffle=True,
        )

        population_rate = raw_frame[TARGET_COLUMN].mean()
        # The whole point: the holdout must resemble the population being scored.
        assert abs(test_df[TARGET_COLUMN].mean() - population_rate) < 0.03
        assert abs(train_df[TARGET_COLUMN].mean() - population_rate) < 0.03

    def test_tenure_remains_a_model_feature(self, raw_frame):
        """Excluding it from the split must not remove it as a predictor."""
        assert "Tenure Months" in build_features(raw_frame.head(50)).columns


class TestTrainServeEquality:
    """
    training_features(x) == serving_features(x).

    The single most important structural property in the codebase; the geography
    change touches the shared builder, so it is re-verified here.
    """

    def test_paths_produce_identical_frames(self, raw_frame):
        from churn_system.training.steps.data_validation import run_data_validation
        from churn_system.training.steps.feature_engineering import (
            run_feature_engineering,
        )

        validated = run_data_validation(raw_frame.head(200).copy())

        training_features = run_feature_engineering(validated)
        serving_features = build_features(validated.copy(), training=False)

        assert list(training_features.columns) == list(serving_features.columns)
        assert training_features.shape == serving_features.shape
        pd.testing.assert_frame_equal(
            training_features.reset_index(drop=True),
            serving_features.reset_index(drop=True),
        )


def pathlib_read(path: str) -> str:
    from pathlib import Path

    return Path(path).read_text(encoding="utf-8")
