# Inference-Time Validation

Training validation alone is not sufficient for production systems.

At inference time, external clients may send incomplete or malformed data. Without validation, models can silently produce incorrect predictions.

## Data Contract

The system enforces the same feature schema used during training.

Rules:
- All required features must exist
- Target column must not be present
- Invalid inputs fail fast

This prevents:
- feature mismatch
- silent prediction corruption
- schema drift errors

Inference validation acts as a safety layer between external inputs and the model.
