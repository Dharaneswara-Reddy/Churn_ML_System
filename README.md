# Churn ML System

## Build → Break → Fix

This repository documents my journey of building a **real-world machine learning system** from scratch and evolving it into a **production-grade ML system**.

The goal of this project is not to achieve the highest accuracy in a notebook, but to understand how machine learning systems behave in practice — including how they fail.

---

## Motivation

Most ML projects stop at training a model and reporting metrics.  
In reality, ML systems fail due to:

- bad or changing data
- schema mismatches
- data leakage
- imbalanced targets
- incorrect assumptions
- fragile pipelines

This project follows a simple but deliberate philosophy:

> **Build the system → Break the system → Fix the system**

Instead of avoiding failures, I intentionally introduce them to understand:
- why they occur,
- how to detect them early,
- and how to design systems that fail safely.

---

## What This Project Is 

### This project **is**:
- an end-to-end ML system
- focused on engineering decisions, not just modeling
- built incrementally from a single `train.py`
- opinionated about correctness, validation, and realism
- documented with reasoning behind every major decision


---

## Problem Statement

Predict customer churn using historical customer data.

The system is designed to:
- ingest raw data,
- validate schema and assumptions,
- train a baseline model,
- evaluate using appropriate metrics for imbalanced data,
- and evolve toward a production-ready pipeline.

---

## Project Evolution

The project intentionally starts simple and evolves over time:

1. **Single script (`train.py`)**
2. Schema enforcement and fail-fast validation
3. Proper evaluation for imbalanced data
4. Pipeline refactoring and modularization
5. Configuration-driven training
6. Model serving and deployment
7. Monitoring, testing, and robustness

At each stage, the system is:
- built,
- broken (intentionally or naturally),
- and fixed with better design.

---

## Key Engineering Principles Followed

- **Fail fast over silent failure**
- **Data contracts over assumptions**
- **Causality over correlation**
- **Insight ≠ feature**
- **Systems over scripts**
- **Documentation of “why”, not just “what”**

---

## Repository Structure (Evolving)

This repository will evolve over time as the system grows.  
The structure reflects the current stage of the project and will change as new requirements emerge.

Key folders include:
- `data/` — raw and processed datasets
- `src/` — core system logic
- `docs/` — design decisions and reasoning
- `model/` — trained model artifacts
- `api/` — inference and serving logic (later stage)

The structure is shaped by real needs, not predefined templates.

---

## Documentation

Important design decisions are documented explicitly in the `docs/` directory.

Examples:
- feature selection and leakage prevention
- schema enforcement rationale
- metric selection for imbalanced data
- trade-offs made during system evolution

This documentation exists to make decisions **defensible and explainable**, especially in production contexts.

---

## Intended Audience

- ML engineers learning system design
- SDEs transitioning into ML engineering
- Anyone who wants to understand how ML systems fail in practice

---

## Status

This project is actively evolving.

Each stage is intentionally incomplete until the next failure exposes what needs to be fixed.

---

## Closing Note

The focus of this project is not to look impressive, but to be **honest**.

A system that works once is easy.  
A system that fails loudly, predictably, and recoverably is engineering.

This repository is an attempt to learn and demonstrate that difference.
