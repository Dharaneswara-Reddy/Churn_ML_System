# Shadow Logging

Every prediction request is future training data.

When users call our API, they are giving us:

1. real feature distributions  
2. real-world edge cases  
3. future drift signals  

So instead of throwing inputs away, we capture them safely.  
This is called **shadow logging / prediction capture / online data collection**.

---

## Why Shadow Logging is Needed

Training datasets are static snapshots of the past, but production data continuously changes.

A model that performs well during training can slowly degrade because:

- customer behavior changes
- feature distributions shift
- new unseen combinations appear
- business policies evolve

Without collecting production inputs, the system has no way to understand how reality is changing.

Shadow logging allows the system to observe production behavior without affecting predictions.

---

## What We Store

For every inference request, the system stores:

- validated input features (not raw payload)
- predicted probability
- final prediction decision
- timestamp (UTC)

The data is appended to:

data/inference_logs/predictions.csv


Only validated features are stored to ensure schema consistency for future retraining.

---

## Important Design Decision

We log **post-validation data**, not raw API input.

Reason:

Raw inputs may contain:
- extra columns
- missing fields
- malformed values

By logging validated data, we guarantee that the collected dataset already satisfies the training schema, making it immediately usable for monitoring or retraining pipelines.

---

## Bug Discovered During Implementation

While implementing shadow logging, a critical issue appeared.

When multiple prediction requests were made:

- rows were appended correctly,
- but column positions changed between rows.

Example:

Row 1 and Row 2 had the same data but in different column orders.

This happened because pandas does not guarantee column ordering when dataframes are constructed dynamically from API payloads. When appending to a CSV file, pandas writes columns exactly in the dataframe order at that moment, causing schema corruption.

This is a dangerous production bug because:

- datasets become unusable for retraining
- features no longer align correctly
- silent data corruption occurs

---

## Fix Implemented

A fixed logging schema was introduced.

Before writing to disk:

1. A deterministic column order is defined using the training schema.
2. The dataframe is reindexed to match this order.
3. Predictions and timestamps are appended afterward.

This guarantees that every logged row follows the exact same structure.

Additionally, timestamps were changed to timezone-aware UTC:

datetime.now(timezone.utc)


to avoid ambiguity across systems and deployments.

---

## Result

The system now continuously builds a clean production dataset while serving predictions.

This enables future capabilities such as:

- data drift detection
- rolling retraining
- production monitoring
- model performance auditing

Shadow logging transforms the system from a static ML model into a learning system that evolves with real-world data.
