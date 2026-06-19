# `churn_system.validation` — Schema-Driven Data Validation

> **Location**: `src/churn_system/validation/`
> **Files**: `validator.py`, `schemas/v1.yaml`

---

## Overview

The `validation` package enforces data quality using **Pandera**, a statistical
data validation library. Instead of writing validation checks in Python code,
constraints are declared in a YAML schema file (`v1.yaml`). The validator loads
this YAML, builds a Pandera `DataFrameSchema`, and validates DataFrames against
it — catching type errors, missing values, and out-of-range values before they
reach the model.

---

## File: `validator.py`

**Purpose**: Translates YAML schema definitions into Pandera validation schemas
and validates DataFrames.

### Constant: `_TYPE_MAP`

Maps YAML type names to Pandera types:

| YAML Type | Pandera Type |
|-----------|-------------|
| `"str"` | `pa.String` |
| `"int"` | `pa.Int` |
| `"float"` | `pa.Float` |
| `"bool"` | `pa.Bool` |

### Function: `load_schema_yaml(path) → dict`

- Reads a YAML file and returns the parsed dictionary.
- **Used by**: `build_pandera_schema()`.

### Function: `build_pandera_schema(schema_dict) → DataFrameSchema`

1. Iterates over the `columns` section of the YAML schema.
2. For each column, creates a Pandera `Column` with:
   - The mapped type (from `_TYPE_MAP`)
   - `required` flag (whether the column must be present)
   - `nullable` flag (whether `NaN`/`None` is allowed)
3. Processes the `checks` section — if a column has an `allowed` list, adds
   a `Check.isin(allowed_values)` constraint (e.g., `Gender` must be in
   `{Male, Female}`).
4. Returns a `DataFrameSchema` with `coerce=True` (auto-converts types where
   possible) and `strict=False` (extra columns are allowed).

### Function: `validate_dataframe(df, *, schema_path) → DataFrame`

- Loads the YAML schema from `schema_path`.
- Builds the Pandera schema.
- Validates the DataFrame with `lazy=True` (collects all errors before raising,
  rather than failing on the first one).
- Returns the validated (and potentially coerced) DataFrame.

**Used by**: `training/steps/data_validation.py` during model training.

---

## Schema File: `schemas/v1.yaml`

The YAML schema defines:
- **29 columns** with their types, required status, and nullable status.
- **Checks** for columns like `Gender` (allowed values: `Male`, `Female`).

Example structure:
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
