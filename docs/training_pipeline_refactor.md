# Training Pipeline Refactor

## Context
The project initially trained models using a single `train.py` file containing
data loading, preprocessing, training, evaluation, and saving logic together.

## Problem
- Tight coupling between steps
- Difficult debugging
- Impossible to reuse components
- No clear pipeline boundaries

## Solution
Training was decomposed into independent steps:

- data_ingestion
- data_validation
- feature_engineering
- model_training
- model_evaluation

Each step now has a single responsibility.

## Result
- Easier experimentation
- Step-level debugging
- Production-like pipeline structure
- Foundation for orchestration