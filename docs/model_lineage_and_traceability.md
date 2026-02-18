# Model Lineage and Traceability

## Context

As the system evolved from simple model training into an automated ML lifecycle,
multiple models started being created through retraining and promotion.
At this stage, knowing *which model is running* was no longer sufficient.
We also needed to know *why* a model exists.

## Problem

Without lineage tracking:

- retrained models overwrite previous ones silently
- performance regressions cannot be traced
- it becomes impossible to identify which dataset produced a model
- debugging production failures becomes guesswork

Models existed, but their history was lost.

## Solution

A lineage tracking mechanism was introduced.

Every promoted model now records:

- model version
- promotion timestamp
- dataset used for training
- retraining trigger
- parent production model
- evaluation metrics

The lineage is stored in:

models/lineage/lineage.json


## Design Principle

Promotion is treated as a lifecycle event, not a file copy.

Each promotion appends a permanent record instead of replacing history.

## Impact

The system can now answer:

- Which retraining introduced a regression?
- What data caused performance change?
- What was the previous production model?

This enables auditability and safe evolution of the ML system.
