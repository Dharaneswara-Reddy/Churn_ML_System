# Model Versioning and Promotion

Training a model does not automatically make it production-ready.

To separate experimentation from deployment, models are stored in two environments:

## Experiments

models/experiments/
churn_model_v1/
churn_model_v2/



Each training run produces a versioned artifact containing:
- model.pkl
- metadata.json

These models are evaluated but never directly served.

## Production

models/production/current

The API only loads models from the production directory. 

## Promotion Process

promotion copies a selected experiment into production using: 

python -m churn_system.promote <version>


This ensures:
- controlled releases
- easy rollback
- reproducible deployments



