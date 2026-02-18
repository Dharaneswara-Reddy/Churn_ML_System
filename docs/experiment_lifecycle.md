# Experiment Lifecycle

## Overview

The system follows a structured lifecycle for models:

1. Training creates experiment versions.
2. Monitoring detects data drift.
3. Retraining prepares updated datasets.
4. Challenger models are evaluated.
5. Promotion replaces the production model.
6. Lineage records the decision permanently.

## Lifecycle Flow

Training
↓
Experiment Model
↓
Monitoring & Drift Detection
↓
Retraining Trigger
↓
Champion vs Challenger Evaluation
↓
Promotion
↓
Lineage Recording



## Key Idea

Deployment is not automatic.
A retrained model must outperform the production model before promotion.

## Result

The system evolves safely while maintaining historical traceability.

