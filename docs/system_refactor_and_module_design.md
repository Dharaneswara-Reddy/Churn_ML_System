# System Refactor and Module Design

## Context

Initially the project started as a single training script.
As monitoring, retraining, API serving, and promotion were added,
the codebase became difficult to reason about.

A structural refactor was required before complexity increased further.

## Design Decision

The system was reorganized based on responsibility instead of file type.

api/ → request handling
training/ → model building
monitoring/ → drift & health checks
lifecycle/ → orchestration and promotion
inference/ → prediction logic
logging/ → observability
config/ → environment configuration
new_data/ → dataset construction logic


## Reasoning

Grouping by responsibility allows:

- independent evolution of subsystems
- easier debugging
- clearer ownership boundaries
- production-like architecture

## Outcome

The project transitioned from a script-based workflow into a modular ML system
where lifecycle stages are clearly separated.
