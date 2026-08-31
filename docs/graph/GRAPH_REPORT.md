# Graph Report - .  (2026-08-31)

## Corpus Check
- 143 files · ~128,549 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1659 nodes · 2765 edges · 65 communities detected
- Extraction: 66% EXTRACTED · 34% INFERRED · 0% AMBIGUOUS · INFERRED: 936 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Test Fixtures & Isolation|Test Fixtures & Isolation]]
- [[_COMMUNITY_FastAPI Request Layer|FastAPI Request Layer]]
- [[_COMMUNITY_Dynamic Schema Generation|Dynamic Schema Generation]]
- [[_COMMUNITY_Inference & Promotion Gates|Inference & Promotion Gates]]
- [[_COMMUNITY_Artifact Signing & Bundle Swap|Artifact Signing & Bundle Swap]]
- [[_COMMUNITY_Distributed Leader Election|Distributed Leader Election]]
- [[_COMMUNITY_Explainability & Readiness Probes|Explainability & Readiness Probes]]
- [[_COMMUNITY_Model Registry Hot-Reload|Model Registry Hot-Reload]]
- [[_COMMUNITY_Data Validation & Event Store|Data Validation & Event Store]]
- [[_COMMUNITY_Probability Calibration|Probability Calibration]]
- [[_COMMUNITY_Architecture Diagrams|Architecture Diagrams]]
- [[_COMMUNITY_Drift Detection (PSI)|Drift Detection (PSI)]]
- [[_COMMUNITY_Deprecated Field Compatibility|Deprecated Field Compatibility]]
- [[_COMMUNITY_Prediction Quality Metrics|Prediction Quality Metrics]]
- [[_COMMUNITY_Feature Builder (TrainServe)|Feature Builder (Train/Serve)]]
- [[_COMMUNITY_Model Lineage & Promotion|Model Lineage & Promotion]]
- [[_COMMUNITY_JSONB Migration|JSONB Migration]]
- [[_COMMUNITY_Concurrent Candidate Training|Concurrent Candidate Training]]
- [[_COMMUNITY_Feature Schema Comparison|Feature Schema Comparison]]
- [[_COMMUNITY_Scaling & Kubernetes Rationale|Scaling & Kubernetes Rationale]]
- [[_COMMUNITY_State Metrics Publishing|State Metrics Publishing]]
- [[_COMMUNITY_Training Data Schema|Training Data Schema]]
- [[_COMMUNITY_Serving Concurrency Rationale|Serving Concurrency Rationale]]
- [[_COMMUNITY_Alembic Env & Split History|Alembic Env & Split History]]
- [[_COMMUNITY_Python Dependencies|Python Dependencies]]
- [[_COMMUNITY_AWS Deployment Runbook|AWS Deployment Runbook]]
- [[_COMMUNITY_Structured Logging|Structured Logging]]
- [[_COMMUNITY_Outbox Status Migration|Outbox Status Migration]]
- [[_COMMUNITY_GitNexus Agent Integration|GitNexus Agent Integration]]
- [[_COMMUNITY_Package Initialisers|Package Initialisers]]
- [[_COMMUNITY_Event Store Baseline Migration|Event Store Baseline Migration]]
- [[_COMMUNITY_Labels & Lease Migration|Labels & Lease Migration]]
- [[_COMMUNITY_ChampionChallenger Policy|Champion/Challenger Policy]]
- [[_COMMUNITY_Prometheus Alert Rules|Prometheus Alert Rules]]
- [[_COMMUNITY_API Smoke Script|API Smoke Script]]
- [[_COMMUNITY_Evaluation Metric Choice|Evaluation Metric Choice]]
- [[_COMMUNITY_Drift Thresholds|Drift Thresholds]]
- [[_COMMUNITY_Lifecycle Sampling Rationale|Lifecycle Sampling Rationale]]
- [[_COMMUNITY_Repo Structure & Testing|Repo Structure & Testing]]
- [[_COMMUNITY_Singleton Double-Check Lock|Singleton Double-Check Lock]]
- [[_COMMUNITY_Registry Reset Helper|Registry Reset Helper]]
- [[_COMMUNITY_Contract Cache Invalidation|Contract Cache Invalidation]]
- [[_COMMUNITY_ChampionChallenger Fixture|Champion/Challenger Fixture]]
- [[_COMMUNITY_Scheduler Interval Guard|Scheduler Interval Guard]]
- [[_COMMUNITY_Repository Commands|Repository Commands]]
- [[_COMMUNITY_Raw Feature Spelling|Raw Feature Spelling]]
- [[_COMMUNITY_Executive Summary|Executive Summary]]
- [[_COMMUNITY_System Architecture Overview|System Architecture Overview]]
- [[_COMMUNITY_Tech Stack|Tech Stack]]
- [[_COMMUNITY_Quick Start|Quick Start]]
- [[_COMMUNITY_API Reference|API Reference]]
- [[_COMMUNITY_Endpoint Overview|Endpoint Overview]]
- [[_COMMUNITY_Vulnerability Reporting|Vulnerability Reporting]]
- [[_COMMUNITY_Global Random Seed|Global Random Seed]]
- [[_COMMUNITY_Model Version Constant|Model Version Constant]]
- [[_COMMUNITY_SQLAlchemy Engine|SQLAlchemy Engine]]
- [[_COMMUNITY_Session Factory|Session Factory]]
- [[_COMMUNITY_PSI Threshold Constant|PSI Threshold Constant]]
- [[_COMMUNITY_Drift Feature Limit|Drift Feature Limit]]
- [[_COMMUNITY_Target Column Constant|Target Column Constant]]
- [[_COMMUNITY_Artifact Path Helpers|Artifact Path Helpers]]
- [[_COMMUNITY_Error Response Contract|Error Response Contract]]
- [[_COMMUNITY_Candidate Model Registry|Candidate Model Registry]]
- [[_COMMUNITY_Preprocessing Transformer|Preprocessing Transformer]]
- [[_COMMUNITY_Logger Factory|Logger Factory]]

## God Nodes (most connected - your core abstractions)
1. `ModelRegistry` - 90 edges
2. `OutboxEvent` - 89 edges
3. `OutboxStatus` - 62 edges
4. `ErrorBody` - 45 edges
5. `PredictionEvent` - 40 edges
6. `Base` - 32 edges
7. `ArtifactSignatureError` - 24 edges
8. `ReadWriteLock` - 24 edges
9. `main()` - 22 edges
10. `verify_bundle_signature()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `predict_batch()` --calls--> `build_features(df, training=False)`  [INFERRED]
  /home/toji339/Documents/Churn_Ml_System/src/churn_system/api/api.py → docs/features.md
- `YAML Base + Env Overrides Config Strategy` --conceptually_related_to--> `load_config()`  [INFERRED]
  CLAUDE.md → /home/toji339/Documents/Churn_Ml_System/src/churn_system/config/config.py
- `predict()` --calls--> `build_features(df, training=False)`  [EXTRACTED]
  /home/toji339/Documents/Churn_Ml_System/src/churn_system/api/api.py → docs/features.md
- `predict()` --calls--> `validate_inference_data(df)`  [EXTRACTED]
  /home/toji339/Documents/Churn_Ml_System/src/churn_system/api/api.py → docs/root_modules.md
- `predict()` --references--> `ErrorBody(BaseModel)`  [EXTRACTED]
  /home/toji339/Documents/Churn_Ml_System/src/churn_system/api/api.py → docs/api.md

## Hyperedges (group relationships)
- **Self-Healing MLOps Loop** — readme_gateway_component, readme_model_registry_component, readme_drift_engine_component, readme_retrain_pipeline_component, readme_lifecycle_manager_component [EXTRACTED 0.90]
- **Model Bundle Contract Enforcement** — claude_model_bundle_contract, api_generate_request_model, root_modules_validate_model_bundle, lifecycle_promote_schemas_match, root_modules_validate_inference_data [INFERRED 0.85]
- **Leakage & PII Exclusion Discipline** — readme_target_leakage_rationale, readme_geography_exclusion_rationale, features_drop_columns_constant, security_known_exposure_geo_history, events_sensitive_keys [INFERRED 0.85]
- **Post-Promotion Hot-Reload Flow** — okflifecycle_promotion, okfserving_hot_reload_trigger, okfinference_cache_invalidation, okfexplainability_reset_explainer [INFERRED 0.85]
- **Transactional Outbox Delivery Flow** — okfevents_write_flow, okfevents_outbox_event, okfworkers_outbox_worker, okfworkers_skip_locked [INFERRED 0.85]
- **Multi-Replica Scaling Requirements** — scaling_horizontal_scaling, scaling_postgres_vs_sqlite, scaling_shared_model_volume, deploykubernetesreadme_scaling_table [INFERRED 0.80]
- **Training Pipeline Execution Flow** — train_start, set_global_seeds, configure_mlflow, data_ingestion, data_validation, feature_engineering, train_test_split, save_training_reference, model_training, model_evaluation, select_winner, save_artifacts, log_mlflow, train_done [INFERRED 0.85]
- **API Predict Request Flow** — client, shutdown_middleware, rate_limiter, api_key_check, pydantic_validation, feature_engineering, inference_schema_validation, inference_runner, apply_threshold, prediction_events, prometheus_metrics, json_response [INFERRED 0.85]
- **Lifecycle Orchestration Flow** — scheduler, orchestrator, model_health, drift_check, build_retraining_dataset, train_start, champion_vs_challenger, schema_compatible, challenger_beats_champion, model_promotion, keep_current_champion, model_healthy, lineage_tracker, auto_rollback, cycle_complete [INFERRED 0.85]

## Communities

### Community 0 - "Test Fixtures & Isolation"
Cohesion: 0.03
Nodes (129): _isolate_event_store(), isolated_paths(), propagating_logger(), Shared test configuration and isolation fixtures.  Two kinds of setup live here:, A CSV satisfying the full raw column contract.      Uses the real dataset when i, Let ``caplog`` observe project loggers.      ``get_logger`` sets ``propagate = F, Reset process-global singletons and caches around every test., Empty the prediction/outbox tables between tests.      The engine is built once (+121 more)

### Community 1 - "FastAPI Request Layer"
Cohesion: 0.02
Nodes (141): _auth_enabled(), body_size_middleware(), _build_request_model(), _chain_sigterm(), _check_key(), erase_subject(), ErrorBody(BaseModel), explain() (+133 more)

### Community 2 - "Dynamic Schema Generation"
Cohesion: 0.02
Nodes (120): generate_request_model(), _load_feature_types_from_reference(feature_schema), schema_generator.py Module, Rationale: Import-Time Config Drives Test Conventions, YAML Base + Env Overrides Config Strategy, Per-Package Docs + okf/ Knowledge Bundle Practice, Experiment Version Discovery By Directory Name, Config Resolved At Import Time (+112 more)

### Community 3 - "Inference & Promotion Gates"
Cohesion: 0.03
Nodes (68): Exception, Inference Pipeline  Handles prediction workflow used by API layer., Execute inference workflow., run_inference_pipeline(), compare_models(), evaluate_promotion_gates(), get_latest_experiment(), _load_confidence_intervals() (+60 more)

### Community 4 - "Artifact Signing & Bundle Swap"
Cohesion: 0.04
Nodes (70): ArtifactSignatureError, _bundle_swap_lock(), _canonical_metadata_bytes(), _cfg(), _compute_and_write_signature(), compute_bundle_digest(), experiment_dir(), experiments_dir() (+62 more)

### Community 5 - "Distributed Leader Election"
Cohesion: 0.03
Nodes (51): advisory_lock_key(), backend_name(), elect_leader(), _event_store_url(), _file_lock(), leader_lock(), _lock_path(), _postgres_advisory_lock() (+43 more)

### Community 6 - "Explainability & Readiness Probes"
Cohesion: 0.03
Nodes (72): Liveness/Readiness Probes, Why /ready Does Real Work, KernelExplainer, SHAP Explainer, Explainer Thread Safety, TreeExplainer, Why Explainability Matters, API-Key Authentication (+64 more)

### Community 7 - "Model Registry Hot-Reload"
Cohesion: 0.03
Nodes (47): _env_flag(), Read a boolean env var without the classic ``"0" is truthy`` trap.      ``bool(o, Hot-reload the production model (thread-safe write).          The bundle is veri, api_module(), _patch_model_contract(), Regression: /predict happy path with mocked model.  Tests monkeypatch ``_get_mod, Return a lightweight stub that mimics sklearn predict_proba()., Patch model contract loader so api.py can be imported and executed without real (+39 more)

### Community 8 - "Data Validation & Event Store"
Cohesion: 0.04
Nodes (45): Data Validation Step  Ensures dataset satisfies schema and training requirements, Validate training dataset before feature engineering., run_data_validation(), Prediction storage adapter.  P1: durable storage in DB outbox (replaces CSV appe, store_prediction(), load_labeled_events(), purge_subject(), record_label() (+37 more)

### Community 9 - "Probability Calibration"
Cohesion: 0.04
Nodes (53): expected_calibration_error(), measure_calibration(), needs_calibration(), Calibration measurement.  A model can rank well and still be badly calibrated. T, Describe how well predicted probabilities match observed frequencies.      Must, Decide whether a calibration correction is warranted.      True when the mean pr, Bin predictions by predicted probability and report the actual rate in each., Weighted mean absolute gap between predicted and actual rate per bin. (+45 more)

### Community 10 - "Architecture Diagrams"
Cohesion: 0.04
Nodes (71): All Modules (api/, training/, monitoring/, lifecycle/, events/, inference/), API Key Check (X-API-Key header), Apply Threshold (default 0.5), Artifact Validator (bundle checks), Auto Rollback (lineage-based), Build Retraining Dataset (original + prod logs), Calculate PSI per Feature (10-bin histogram), Challenger Beats Champion? (decision) (+63 more)

### Community 11 - "Drift Detection (PSI)"
Cohesion: 0.04
Nodes (42): calculate_psi(), detect_drift(), min_production_samples(), psi_bins(), psi_threshold(), Data Drift Detection Module  Compares training data distribution with production, Compare training and production datasets and report feature-level drift., Compute Population Stability Index (PSI).      Parameters     ----------     exp (+34 more)

### Community 12 - "Deprecated Field Compatibility"
Cohesion: 0.05
Nodes (39): _deprecated_fields(), _payload_to_row(), _probe_row(), Turn a validated request into a feature row, dropping deprecated fields.      Th, Deprecated field names for the *current* request model.      Recomputed rather t, One realistic feature row for the readiness probe.      Taken from the training, deprecated_request_fields(), generate_request_model() (+31 more)

### Community 13 - "Prediction Quality Metrics"
Cohesion: 0.05
Nodes (40): compute_class_balance(), compute_confidence_distribution(), compute_expected_calibration_error(), compute_gini_coefficient(), compute_prediction_entropy(), generate_calibration_report(), Prediction Calibration and Confidence Monitor.  Industry-level monitoring of mod, Analyze the distribution of prediction confidence scores.      Returns summary s (+32 more)

### Community 14 - "Feature Builder (Train/Serve)"
Cohesion: 0.07
Nodes (26): build_features(), Feature Builder  Single source of truth for feature preparation. Used by BOTH tr, Prepare model-ready features.      Parameters     ----------     df : pd.DataFra, Feature Engineering Step  Builds model-ready features using shared feature build, Transform validated dataset into model features., run_feature_engineering(), Tests for the shared feature builder., Create a minimal raw dataframe matching expected schema. (+18 more)

### Community 15 - "Model Lineage & Promotion"
Cohesion: 0.08
Nodes (23): load_lineage(), Model Lineage Tracking.   Maintains history of all promoted models., Append a lineage record for a promoted model., record_lineage(), save_lineage(), promote_model(), Promotion of a trained experiment bundle into the production serving slot., Describe a schema change in the terms an operator has to reason about. (+15 more)

### Community 16 - "JSONB Migration"
Cohesion: 0.09
Nodes (15): downgrade(), _is_postgresql(), Store JSON payloads as JSONB on PostgreSQL.  Revision ID: 0004 Revises: 0003 Cre, upgrade(), Migration mechanics that are backend-independent.  Live PostgreSQL behaviour (JS, A migration without a real ``downgrade`` cannot be rolled back, which turns, If alembic.ini carried its own URL, ``alembic upgrade head`` could migrate, Guards against a migration being deleted rather than superseded. (+7 more)

### Community 17 - "Concurrent Candidate Training"
Cohesion: 0.11
Nodes (16): build_preprocessor(), get_model_registry(), Concurrent Model Training  Trains candidate models in PARALLEL using ThreadPoolE, Defines candidate models for competition., Train a single candidate model. Designed to be submitted to a thread pool., Train all registered candidates CONCURRENTLY and return name -> fitted Pipeline., train_candidate_models(), _train_single_candidate() (+8 more)

### Community 18 - "Feature Schema Comparison"
Cohesion: 0.12
Nodes (14): compare_feature_schemas(), load_schema(), Compare production and challenger feature schemas., added_feature_metadata(), identical_metadata(), prod_metadata(), Unit tests for schema comparison and feature schema validation., Create a production metadata file. (+6 more)

### Community 19 - "Scaling & Kubernetes Rationale"
Cohesion: 0.14
Nodes (16): ReadWriteMany Storage Rationale, Kubernetes Horizontal Scalability Table, Scheduler Replicas=1 Recreate Strategy, Deployment Shapes, AWS Deployment Known Limits, Batch Chunking, Why Chunking Was Removed, Event Store Cost Comparison (+8 more)

### Community 20 - "State Metrics Publishing"
Cohesion: 0.15
Nodes (14): drift_report_age_seconds(), _publish_champion(), _publish_drift(), _publish_outbox_backlog(), _publish_prediction_rates(), Publish persisted lifecycle state as Prometheus metrics from the API process.  T, Publish outbox depth by status.      Dead-lettered events used to be indistingui, Refresh every gauge derived from persisted state.      Called from the ``/metric (+6 more)

### Community 21 - "Training Data Schema"
Cohesion: 0.18
Nodes (8): Schema contracts for training data and inference validation., validate_training_data(), _make_valid_df(), Unit tests for data validation (Pandera schema enforcement)., Create a minimal valid training dataframe., Tests for training data schema enforcement., Sanity check that REQUIRED_COLUMNS contains expected columns., TestValidateTrainingData

### Community 22 - "Serving Concurrency Rationale"
Cohesion: 0.13
Nodes (15): load_model_contract() lru_cache Imported By Value, ModelRegistry.reset() Test Convention, Hand-Rolled ReadWriteLock, Rationale: Many Readers, Exclusive Writer For Hot-Swap, Serving: Singleton Registry With Read-Write Lock, AWS Single-Node Deployment, Containerized Deployment (Docker Compose), Environment Configuration Options Table (+7 more)

### Community 23 - "Alembic Env & Split History"
Cohesion: 0.14
Nodes (13): Temporal Split Sorted By Tenure Months, Rationale: Deliberately Not Random, _database_url(), Alembic environment.  Pulls the database URL from the application's own configur, Emit SQL to stdout without connecting., Run migrations against a live connection., run_migrations_offline(), run_migrations_online() (+5 more)

### Community 24 - "Python Dependencies"
Cohesion: 0.2
Nodes (10): Python Dependencies List, FastAPI, NumPy, Pandas, Pydantic v2, PyYAML, Requests, scikit-learn (+2 more)

### Community 25 - "AWS Deployment Runbook"
Cohesion: 0.22
Nodes (9): Read-Only Bundle Mount, AWS Single Node Deployment, Deploy Procedure, No SSH, Session Manager Access, Pinned Commit Tag Rationale, Why Scheduler Is Not Running, SSM SecureString Secrets, Signed Model Bundle (+1 more)

### Community 26 - "Structured Logging"
Cohesion: 0.33
Nodes (5): get_logger(), JSONFormatter, Central logging configuration for the churn system.  Supports: - JSON-structured, Emit each log record as a single-line JSON object., Create and return a configured logger.      Parameters     ----------     name :

### Community 27 - "Outbox Status Migration"
Cohesion: 0.53
Nodes (5): _columns(), downgrade(), _indexes(), Add explicit outbox status, dead-letter reason, and retention indexes.  Revision, upgrade()

### Community 28 - "GitNexus Agent Integration"
Cohesion: 0.33
Nodes (6): GitNexus MCP Integration (AGENTS.md), GitNexus Graph Schema, GitNexus Resources Reference Table, GitNexus Skills Reference Table, GitNexus Tools Reference Table, GitNexus MCP Integration (CLAUDE.md)

### Community 29 - "Package Initialisers"
Cohesion: 0.4
Nodes (1): Explainability package for model interpretation using SHAP.

### Community 30 - "Event Store Baseline Migration"
Cohesion: 0.5
Nodes (3): _has_table(), Event store baseline: prediction and outbox tables.  Revision ID: 0001 Revises:, upgrade()

### Community 31 - "Labels & Lease Migration"
Cohesion: 0.5
Nodes (3): _columns(), Add ground-truth labels, subject keys, and outbox lease columns.  Revision ID: 0, upgrade()

### Community 32 - "Champion/Challenger Policy"
Cohesion: 0.5
Nodes (4): Rationale: Atomic Directory-Rename Swap Prevents Missing Model, Safe Champion vs Challenger Promotion Steps, Ground Truth Feedback Loop, Automated Rollback Policy

### Community 33 - "Prometheus Alert Rules"
Cohesion: 0.5
Nodes (4): Alert Rules (observability.md), Pre-configured Prometheus Alert Rules, Note On Metric Scope (Pushgateway Requirement), Observability & Monitoring (Prometheus Metrics)

### Community 34 - "API Smoke Script"
Cohesion: 0.67
Nodes (1): Manual smoke check against a running API (uvicorn churn_system.api.api:app).  Us

### Community 36 - "Evaluation Metric Choice"
Cohesion: 1.0
Nodes (2): Model Evaluation Metrics, Rationale: ROC-AUC Deliberately Not The Headline Metric

### Community 37 - "Drift Thresholds"
Cohesion: 1.0
Nodes (2): Drift Threshold Spectrum, PSI Formula

### Community 38 - "Lifecycle Sampling Rationale"
Cohesion: 1.0
Nodes (2): Rationale: Minimum Production Sample Size Guards Against Spurious Retraining, MLOps Lifecycle & Drift Detection

### Community 39 - "Repo Structure & Testing"
Cohesion: 1.0
Nodes (2): Repository Structure, Testing & Quality Assurance

### Community 44 - "Singleton Double-Check Lock"
Cohesion: 1.0
Nodes (1): Thread-safe singleton access (double-checked locking pattern).          The oute

### Community 45 - "Registry Reset Helper"
Cohesion: 1.0
Nodes (1): Reset the singleton (used in tests).

### Community 46 - "Contract Cache Invalidation"
Cohesion: 1.0
Nodes (1): Drop caches keyed to the previous model.

### Community 47 - "Champion/Challenger Fixture"
Cohesion: 1.0
Nodes (1): Build a champion/challenger pair.          Metrics are padded with recall/precis

### Community 48 - "Scheduler Interval Guard"
Cohesion: 1.0
Nodes (1): 0 spins the retrain loop; negative crashes time.sleep, and with         restart:

### Community 49 - "Repository Commands"
Cohesion: 1.0
Nodes (1): Repository Commands Reference

### Community 50 - "Raw Feature Spelling"
Cohesion: 1.0
Nodes (1): Raw-Dataset Feature Spelling Convention

### Community 51 - "Executive Summary"
Cohesion: 1.0
Nodes (1): Executive Summary

### Community 52 - "System Architecture Overview"
Cohesion: 1.0
Nodes (1): System Architecture Diagram

### Community 53 - "Tech Stack"
Cohesion: 1.0
Nodes (1): Tech Stack & Enterprise Components Table

### Community 54 - "Quick Start"
Cohesion: 1.0
Nodes (1): Quick Start Guide

### Community 55 - "API Reference"
Cohesion: 1.0
Nodes (1): API Reference & Usage

### Community 56 - "Endpoint Overview"
Cohesion: 1.0
Nodes (1): Endpoints Overview Table

### Community 57 - "Vulnerability Reporting"
Cohesion: 1.0
Nodes (1): Vulnerability Reporting Process

### Community 58 - "Global Random Seed"
Cohesion: 1.0
Nodes (1): GLOBAL_SEED = 42

### Community 59 - "Model Version Constant"
Cohesion: 1.0
Nodes (1): MODEL_VERSION

### Community 60 - "SQLAlchemy Engine"
Cohesion: 1.0
Nodes (1): ENGINE SQLAlchemy Engine

### Community 61 - "Session Factory"
Cohesion: 1.0
Nodes (1): SessionLocal Session Factory

### Community 63 - "PSI Threshold Constant"
Cohesion: 1.0
Nodes (1): PSI_THRESHOLD = 0.2 (drift.py)

### Community 64 - "Drift Feature Limit"
Cohesion: 1.0
Nodes (1): DRIFT_FEATURE_LIMIT = 2

### Community 65 - "Target Column Constant"
Cohesion: 1.0
Nodes (1): TARGET_COLUMN = 'Churn Value' (schema.py)

### Community 66 - "Artifact Path Helpers"
Cohesion: 1.0
Nodes (1): Artifact Path Helper Functions

### Community 68 - "Error Response Contract"
Cohesion: 1.0
Nodes (1): ErrorBody Error Responses

### Community 69 - "Candidate Model Registry"
Cohesion: 1.0
Nodes (1): Candidate Models

### Community 70 - "Preprocessing Transformer"
Cohesion: 1.0
Nodes (1): Preprocessing ColumnTransformer

### Community 71 - "Logger Factory"
Cohesion: 1.0
Nodes (1): Structured Logger (JSON / text)

## Ambiguous Edges - Review These
- `Temporal Split Sorted By Tenure Months` → `Rationale: Stratified Holdout Replaces Flawed Temporal Split`  [AMBIGUOUS]
  CLAUDE.md · relation: conceptually_related_to
- `Model Evaluation Methodology` → `SELECTION_METRIC Constant`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `FastAPI Server (/predict /predict/batch /health /metrics)` → `Model Contract (cached metadata)`  [AMBIGUOUS]
  docs/images/system_architecture.png · relation: calls

## Knowledge Gaps
- **467 isolated node(s):** `Schema contracts for training data and inference validation.`, `Validate inference dataframe against MODEL FEATURE SCHEMA     (not raw dataset s`, `MLflow integration helpers with retry logic for transient network failures.`, `Configure MLflow tracking URI and experiment. Returns config dict.`, `Log and (optionally) register model with MLflow Model Registry.      Retries on` (+462 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Package Initialisers`** (5 nodes): `Explainability package for model interpretation using SHAP.`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `API Smoke Script`** (3 nodes): `smoke_api.py`, `main()`, `Manual smoke check against a running API (uvicorn churn_system.api.api:app).  Us`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Evaluation Metric Choice`** (2 nodes): `Model Evaluation Metrics`, `Rationale: ROC-AUC Deliberately Not The Headline Metric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Drift Thresholds`** (2 nodes): `Drift Threshold Spectrum`, `PSI Formula`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Lifecycle Sampling Rationale`** (2 nodes): `Rationale: Minimum Production Sample Size Guards Against Spurious Retraining`, `MLOps Lifecycle & Drift Detection`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Repo Structure & Testing`** (2 nodes): `Repository Structure`, `Testing & Quality Assurance`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Singleton Double-Check Lock`** (1 nodes): `Thread-safe singleton access (double-checked locking pattern).          The oute`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Registry Reset Helper`** (1 nodes): `Reset the singleton (used in tests).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Contract Cache Invalidation`** (1 nodes): `Drop caches keyed to the previous model.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Champion/Challenger Fixture`** (1 nodes): `Build a champion/challenger pair.          Metrics are padded with recall/precis`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Scheduler Interval Guard`** (1 nodes): `0 spins the retrain loop; negative crashes time.sleep, and with         restart:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Repository Commands`** (1 nodes): `Repository Commands Reference`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Raw Feature Spelling`** (1 nodes): `Raw-Dataset Feature Spelling Convention`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Executive Summary`** (1 nodes): `Executive Summary`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `System Architecture Overview`** (1 nodes): `System Architecture Diagram`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tech Stack`** (1 nodes): `Tech Stack & Enterprise Components Table`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Quick Start`** (1 nodes): `Quick Start Guide`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `API Reference`** (1 nodes): `API Reference & Usage`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Endpoint Overview`** (1 nodes): `Endpoints Overview Table`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Vulnerability Reporting`** (1 nodes): `Vulnerability Reporting Process`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Global Random Seed`** (1 nodes): `GLOBAL_SEED = 42`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Model Version Constant`** (1 nodes): `MODEL_VERSION`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SQLAlchemy Engine`** (1 nodes): `ENGINE SQLAlchemy Engine`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Session Factory`** (1 nodes): `SessionLocal Session Factory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `PSI Threshold Constant`** (1 nodes): `PSI_THRESHOLD = 0.2 (drift.py)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Drift Feature Limit`** (1 nodes): `DRIFT_FEATURE_LIMIT = 2`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Target Column Constant`** (1 nodes): `TARGET_COLUMN = 'Churn Value' (schema.py)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Artifact Path Helpers`** (1 nodes): `Artifact Path Helper Functions`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Error Response Contract`** (1 nodes): `ErrorBody Error Responses`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Candidate Model Registry`** (1 nodes): `Candidate Models`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Preprocessing Transformer`** (1 nodes): `Preprocessing ColumnTransformer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Logger Factory`** (1 nodes): `Structured Logger (JSON / text)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Temporal Split Sorted By Tenure Months` and `Rationale: Stratified Holdout Replaces Flawed Temporal Split`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Model Evaluation Methodology` and `SELECTION_METRIC Constant`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `FastAPI Server (/predict /predict/batch /health /metrics)` and `Model Contract (cached metadata)`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **Why does `main()` connect `Probability Calibration` to `FastAPI Request Layer`, `Inference & Promotion Gates`, `Artifact Signing & Bundle Swap`, `Data Validation & Event Store`, `Feature Builder (Train/Serve)`, `Concurrent Candidate Training`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Why does `OutboxStatus` connect `Test Fixtures & Isolation` to `Data Validation & Event Store`, `Artifact Signing & Bundle Swap`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `predict()` connect `FastAPI Request Layer` to `Dynamic Schema Generation`, `Inference & Promotion Gates`, `Deprecated Field Compatibility`, `Artifact Signing & Bundle Swap`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Are the 80 inferred relationships involving `ModelRegistry` (e.g. with `FeedbackBody` and `Churn prediction HTTP API.  Distributed Systems & Concurrency Features: - Async`) actually correct?**
  _`ModelRegistry` has 80 INFERRED edges - model-reasoned connections that need verification._