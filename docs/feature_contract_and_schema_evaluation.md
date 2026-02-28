# Feature Contract & Schema Evolution

## Problem
Inference validation originally depended on raw dataset columns.
After introducing feature engineering, the model no longer consumed
raw columns, causing API failures.

## Root Cause
Training schema ≠ Inference schema.

The API validated dataset schema instead of model feature schema.

## Solution
Feature schema is now saved inside model metadata:

- feature_schema
- feature_count

Inference validation loads schema from production metadata.

## Impact
- Training-serving skew eliminated
- Safe feature evolution
- Backward-compatible deployments