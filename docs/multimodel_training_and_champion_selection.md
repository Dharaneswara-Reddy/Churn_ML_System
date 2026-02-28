# Multi-Model Training & Champion Selection

## Motivation
Earlier versions trained only Logistic Regression,
limiting experimentation.

## Implementation
Training step now evaluates multiple models:

- Logistic Regression
- Random Forest
- Gradient Boosting

Each model is trained and evaluated automatically.

## Selection Strategy
Champion model selected using ROC-AUC comparison.

## Outcome
- Automatic best-model selection
- Reduced manual experimentation
- Industry-standard training workflow