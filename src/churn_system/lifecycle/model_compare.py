"""
Champion vs Challenger Comparison.

Compares the current production (champion) model with the
latest experiment (challenger) model using evaluation metrics
and feature schema compatibility checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from churn_system.artifacts import latest_experiment_dir, load_metadata
from churn_system.config.config import CONFIG
from churn_system.lifecycle.schema_compare import compare_feature_schemas
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])


def _production_metadata_path() -> Path:
    return Path(CONFIG["paths"]["production_model"]).parent / "metadata.json"


def _promotion_metric() -> str:
    return str(CONFIG.get("model_promotion", {}).get("metric", "pr_auc"))


def _min_improvement() -> float:
    return float(CONFIG.get("model_promotion", {}).get("min_improvement", 0.0))


def _promotion_config() -> dict[str, Any]:
    return dict(CONFIG.get("model_promotion", {}))


def evaluate_promotion_gates(
    champion_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
    challenger_intervals: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Apply every configured promotion gate.

    Returns (passed, reasons). ``reasons`` always explains the decision — an
    operator should never have to diff two metadata files to learn why a model
    was or was not promoted.

    Gates, in the order a bad model is most likely to trip them:

    1. **Absolute floors** — a challenger must be independently acceptable, not
       merely better than a bad champion. A model with recall 0.0 must never ship
       regardless of how it compares.
    2. **Minimum improvement** — must beat the champion by more than measurement
       noise. Bootstrap on this dataset put the PR-AUC 95% CI width near 0.165, so
       the previous 0.0 margin promoted on noise.
    3. **No-regression** — a gain on the gate metric must not come at the cost of a
       collapse elsewhere.
    4. **Statistical significance** — the challenger's CI lower bound must exceed
       the champion's point estimate, so an improvement inside the noise band is
       rejected rather than shipped.
    """
    config = _promotion_config()
    metric = _promotion_metric()
    reasons: list[str] = []
    passed = True

    # --- 1. absolute floors ---------------------------------------------------
    for gate_key, metric_key in (
        ("min_recall", "recall"),
        ("min_precision", "precision"),
        ("min_pr_auc", "pr_auc"),
    ):
        floor = config.get(gate_key)
        if floor is None:
            continue
        value = challenger_metrics.get(metric_key)
        if value is None:
            passed = False
            reasons.append(f"challenger is missing {metric_key!r}, required by {gate_key}")
        elif float(value) < float(floor):
            passed = False
            reasons.append(
                f"{metric_key}={float(value):.4f} is below the {gate_key} floor of {float(floor):.4f}"
            )

    # --- 2. minimum improvement ----------------------------------------------
    champion_score = champion_metrics.get(metric)
    challenger_score = challenger_metrics.get(metric)
    if champion_score is None or challenger_score is None:
        return False, [*reasons, f"metric {metric!r} missing from champion or challenger"]

    improvement = float(challenger_score) - float(champion_score)
    required = _min_improvement()
    if improvement < required:
        passed = False
        reasons.append(
            f"{metric} improvement {improvement:+.4f} is below the required {required:+.4f}"
        )

    # --- 3. no-regression -----------------------------------------------------
    for metric_key, allowance in (config.get("no_regression") or {}).items():
        champion_value = champion_metrics.get(metric_key)
        challenger_value = challenger_metrics.get(metric_key)
        if champion_value is None or challenger_value is None:
            continue
        regression = float(champion_value) - float(challenger_value)
        if regression > float(allowance):
            passed = False
            reasons.append(
                f"{metric_key} regressed by {regression:.4f}, more than the "
                f"{float(allowance):.4f} allowance"
            )

    # --- 4. statistical significance -----------------------------------------
    if config.get("require_statistical_significance"):
        interval = (challenger_intervals or {}).get(metric)
        if not interval or interval.get("lower") is None:
            reasons.append(
                f"no bootstrap interval for {metric!r}; cannot establish significance"
            )
            passed = False
        elif float(interval["lower"]) <= float(champion_score):
            passed = False
            reasons.append(
                f"challenger {metric} CI lower bound {float(interval['lower']):.4f} does not "
                f"exceed the champion's {float(champion_score):.4f} — the improvement is "
                "within measurement noise"
            )

    if passed:
        reasons.append(
            f"all gates passed: {metric} {float(champion_score):.4f} -> "
            f"{float(challenger_score):.4f} ({improvement:+.4f})"
        )
    return passed, reasons


def get_latest_experiment() -> Path | None:
    """Return the newest complete experiment bundle, or None."""
    return latest_experiment_dir()


def load_metrics(path: Path) -> dict[str, Any]:
    """Load evaluation metrics stored inside metadata.json."""
    return load_metadata(path).get("metrics", {})


def compare_models() -> bool:
    """
    Compare the production model with the latest retrained model.

    Decision logic:
        1. Ensure schema compatibility.
        2. Compare the configured promotion metric.
        3. Promote only if the challenger improves on it by at least
           ``model_promotion.min_improvement``.

    The metric and margin come from configuration rather than being hardcoded: the
    training pipeline selects its winner by ``training.selection_metric`` (PR-AUC by
    default, which suits this imbalanced target), so judging promotion by a
    different metric can reject the very model training just chose.
    """

    latest = get_latest_experiment()

    if latest is None:
        logger.warning("No experiment models found.")
        return False

    metric = _promotion_metric()
    min_improvement = _min_improvement()

    challenger_meta = latest / "metadata.json"
    challenger_metrics = load_metrics(challenger_meta)

    production_metadata = _production_metadata_path()

    # First deployment case
    if not production_metadata.exists():
        logger.info("No production model found. Auto-promoting first model.")
        return True

    champion_metrics = load_metrics(production_metadata)

    try:
        schema_report = compare_feature_schemas(production_metadata, challenger_meta)

        logger.info("Schema comparison result: %s", schema_report)

        # BLOCK promotion if breaking change detected
        if schema_report["removed_features"]:
            logger.warning(
                "Breaking schema change detected. "
                "Promotion blocked due to removed features."
            )
            return False

    except Exception:
        logger.exception("Schema comparison failed — refusing to promote")
        return False

    if metric not in champion_metrics or metric not in challenger_metrics:
        logger.error(
            "Promotion metric %r missing (champion=%s, challenger=%s). "
            "Refusing to promote on incomparable metrics.",
            metric,
            sorted(champion_metrics),
            sorted(challenger_metrics),
        )
        return False

    logger.info("--- Champion vs Challenger (%s) ---", metric)
    logger.info("Champion   %s: %.6f", metric, float(champion_metrics[metric]))
    logger.info("Challenger %s: %.6f", metric, float(challenger_metrics[metric]))
    logger.info("Minimum required improvement: %+.6f", min_improvement)

    # Bootstrap intervals are written into experiment_report.json at training time.
    challenger_intervals = _load_confidence_intervals(latest)

    passed, reasons = evaluate_promotion_gates(
        champion_metrics, challenger_metrics, challenger_intervals
    )

    for reason in reasons:
        logger.info("Promotion gate: %s", reason)

    if passed:
        logger.info("Challenger model wins — all promotion gates passed.")
        return True

    logger.info("Champion model retained — promotion gates not satisfied.")
    return False


def _load_confidence_intervals(experiment_dir: Path) -> dict[str, Any]:
    """
    Read the winner's bootstrap intervals from the experiment report.

    Returns an empty mapping when unavailable; the significance gate then refuses
    to promote rather than assuming significance it cannot demonstrate.
    """
    report_path = experiment_dir / "experiment_report.json"
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.exception("Could not read experiment report at %s", report_path)
        return {}

    winner = report.get("winner")
    intervals = report.get("confidence_intervals", {}).get(winner, {})
    if not intervals:
        return {}
    # Stored per-model as a single interval for the selection metric.
    return {report.get("selection_metric", _promotion_metric()): intervals}
