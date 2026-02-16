# System Architecture Overview

This project started as a single training script and gradually evolved into a structured machine learning system.

The goal was not just to train a model, but to design a system that behaves safely and predictably in production.

## Core Components

- Training Pipeline
- Model Artifacts
- Inference Validation
- API Serving Layer
- Configuration Management
- Promotion Workflow
- Logging & Monitoring

The codebase follows a `src/` package layout where all business logic lives inside `churn_system`.


## Design Principle

Build → Break → Fix

Each stage intentionally exposed real-world problems such as data leakage, distribution shift, unsafe inference, and deployment risks, and then introduced engineering solutions to address them.
