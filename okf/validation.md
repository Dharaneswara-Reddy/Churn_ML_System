---
type: Component
title: Data Validation
description: Schema-driven data validation using Pandera from YAML definitions, plus training and inference data contracts enforcing column presence, types, and feature ordering.
tags: [validation, pandera, schema, data-quality]
timestamp: 2026-06-30T00:00:00Z
---

The validation system enforces data quality at two levels: Pandera schema validation from YAML definitions (for rich type and value constraints) and programmatic data contracts (for training completeness and inference feature ordering).

# Pandera Validation from YAML

Instead of writing validation checks in Python, constraints are declared in a YAML schema file (`validation/schemas/v1.yaml`). The validator loads this YAML, builds a Pandera `DataFrameSchema`, and validates DataFrames against it.

## Schema File Structure

The YAML defines 29 columns with types, required status, nullable status, and value checks:

```yaml
columns:
  CustomerID:
    type: str
    required: true
    nullable: false
  Gender:
    type: str
    required: true
    nullable: false
  Tenure Months:
    type: int
    required: true
    nullable: false

checks:
  Gender:
    allowed: ["Male", "Female"]
```

## Schema Building

`build_pandera_schema(schema_dict)` translates YAML into Pandera objects:

1. Maps YAML types to Pandera types (`str` → `pa.String`, `int` → `pa.Int`, `float` → `pa.Float`, `bool` → `pa.Bool`).
2. Creates a `Column` per entry with `required` and `nullable` flags.
3. Adds `Check.isin(allowed_values)` for columns with an `allowed` list.
4. Returns a `DataFrameSchema` with `coerce=True` (auto-converts types) and `strict=False` (extra columns are allowed).

`validate_dataframe(df, *, schema_path)` loads the schema, builds it, and validates with `lazy=True` (collects all errors before raising). Used by the [training pipeline](training.md) during data validation.

# Training Data Contract

`validate_training_data(df)` enforces that raw training data meets expectations:

- All 33 `REQUIRED_COLUMNS` are present in the DataFrame.
- The target column (`Churn Value`) contains only values in `{0, 1}`.
- Raises `ValueError` with descriptive messages on failure.

Constants:
- `TARGET_COLUMN = "Churn Value"`
- `ALLOWED_TARGET_VALUES = {0, 1}`
- `REQUIRED_COLUMNS` — set of 33 column names expected in raw training data

# Inference Data Contract

`validate_inference_data(df)` enforces the model's feature contract at prediction time:

1. Loads the feature schema from the production [model contract](inference.md).
2. Checks that all required model features are present.
3. **Rejects** DataFrames containing the target column (prevents accidental target leakage at inference time).
4. **Reorders columns** to match the exact training-time feature order — critical for models that encode features positionally.

Used by the [API](api.md) and [inference engine](inference.md) before every prediction.
