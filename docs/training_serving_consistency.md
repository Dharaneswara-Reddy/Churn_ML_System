# Training–Serving Consistency

## Risk
Feature transformations differed between training and inference.

## Solution
Single shared feature builder introduced:

features/build_features.py

Used by:
- training pipeline
- inference API

## Outcome
- No feature mismatch
- Stable production predictions