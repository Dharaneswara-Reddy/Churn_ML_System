# Pipeline Orchestration Design

## Objective
Move from script execution to orchestrated ML workflows.

## Pipelines Introduced
- training_pipeline
- inference_pipeline
- monitoring_pipeline

## Responsibilities
Training Pipeline:
- prepares data
- trains models
- saves artifacts

Monitoring Pipeline:
- drift detection
- prediction monitoring
- health reporting

## Result
System behaves like a real ML platform rather than scripts.