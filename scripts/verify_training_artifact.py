"""
Verify that the most recent training run produced a usable model bundle.

A training run exiting 0 only proves it did not crash. This script closes the gap
by doing what serving does: validate the bundle contract, unpickle the model, and
score one row built from the recorded feature schema. It catches a corrupt pickle,
a scikit-learn version mismatch, a metadata/schema disagreement, and a model whose
metrics are no better than chance — none of which change the training exit code.

    python scripts/verify_training_artifact.py [--min-roc-auc 0.6]
"""

from __future__ import annotations

import argparse
import pickle
import sys

import pandas as pd

from churn_system.artifacts import latest_experiment_dir, validate_model_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-roc-auc",
        type=float,
        default=0.55,
        help=(
            "fail if the recorded ROC-AUC is at or below this (default: 0.55). "
            "Lower it for synthetic datasets whose labels carry no signal."
        ),
    )
    args = parser.parse_args()

    experiment = latest_experiment_dir()
    if experiment is None:
        print("FAIL: no experiment bundle found", file=sys.stderr)
        return 1

    model_path = experiment / "model.pkl"
    metadata = validate_model_bundle(model_path)

    schema = metadata["feature_schema"]
    types = metadata.get("feature_types", {})

    with open(model_path, "rb") as handle:
        model = pickle.load(handle)  # noqa: S301 - artifact we just produced

    # A neutral row: zeros for numerics, a placeholder for categoricals. The
    # pipeline's OneHotEncoder uses handle_unknown="ignore", so unseen categories
    # are tolerated and this exercises the real transform chain.
    row = {
        feature: (0 if types.get(feature) in {"int", "float", "bool"} else "unknown")
        for feature in schema
    }
    frame = pd.DataFrame([row])[schema]

    probability = float(model.predict_proba(frame)[:, 1][0])
    if not 0.0 <= probability <= 1.0:
        print(f"FAIL: probability out of range: {probability}", file=sys.stderr)
        return 1

    roc_auc = float(metadata.get("metrics", {}).get("roc_auc", 0.0))
    if roc_auc <= args.min_roc_auc:
        print(
            f"FAIL: roc_auc {roc_auc:.4f} is at or below the {args.min_roc_auc} floor "
            "— the model is no better than chance",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {experiment.name} | {metadata['model_type']} | "
        f"{len(schema)} features | roc_auc={roc_auc:.4f} | probe={probability:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
