"""
Churn prediction HTTP API.

Distributed Systems & Concurrency Features:
- Async endpoints (asyncio) for non-blocking I/O
- Thread-safe ModelRegistry with ReadWriteLock for concurrent reads
- Concurrent batch processing via asyncio.gather + chunked parallelism
- SIGTERM graceful shutdown with connection draining
- Rate limiting and optional API key authentication

Concurrency Model:
- FastAPI runs on an asyncio event loop (single-threaded I/O)
- CPU-bound model inference is offloaded to a thread pool via asyncio.to_thread()
- This prevents blocking the event loop during predict_proba() calls
- Batch requests are chunked and processed concurrently across threads
"""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from churn_system.api.errors import ErrorBody
from churn_system.api.schema_generator import generate_request_model, load_feature_schema
from churn_system.config.config import CONFIG, load_config
from churn_system.events.db import init_db
from churn_system.events.predictions import record_label, store_prediction_event
from churn_system.features.build_features import build_features
from churn_system.logging.logger import get_logger
from churn_system.observability.metrics import (
    EVENT_WRITE_DROPPED_TOTAL,
    EVENT_WRITE_FAILURES_TOTAL,
    EXPLANATION_LATENCY_SECONDS,
    EXPLANATION_REQUESTS_TOTAL,
    INFERENCE_ERRORS_TOTAL,
    LABELS_RECORDED_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    render_latest,
)
from churn_system.schema import validate_inference_data
from churn_system.serving.model_registry import ModelRegistry

logger = get_logger(__name__, CONFIG["logging"]["api"])
config = load_config()


def _env_flag(name: str, default: bool = False) -> bool:
    """
    Read a boolean env var without the classic ``"0" is truthy`` trap.

    ``bool(os.environ.get(name))`` treats "0", "false" and "no" as enabled, which
    is the opposite of what an operator setting CHURN_DISABLE_RATE_LIMIT=0 expects.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


limiter = Limiter(
    key_func=get_remote_address,
    enabled=not _env_flag("CHURN_DISABLE_RATE_LIMIT"),
)

# ---------------------------------------------------------------------------
# Graceful shutdown on SIGTERM (container orchestrator → drain → exit)
# ---------------------------------------------------------------------------
_shutting_down = False

# Dedicated pool for durable event writes. These must not share the default
# executor with inference: a slow event store would otherwise occupy every worker
# thread, and un-awaited writes would queue without bound until `predict` latency
# collapsed under the backlog.
EVENT_WRITER_WORKERS = int(os.environ.get("CHURN_EVENT_WRITER_WORKERS", "4"))
EVENT_WRITER_QUEUE_LIMIT = int(os.environ.get("CHURN_EVENT_WRITER_QUEUE_LIMIT", "1000"))
_event_executor: ThreadPoolExecutor | None = None
_event_backlog = 0
_event_backlog_lock = threading.Lock()


def _handle_sigterm(signum, frame):
    global _shutting_down
    _shutting_down = True
    logger.info("SIGTERM received — draining in-flight requests before shutdown")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Start-up and shutdown wiring.

    The SIGTERM handler is installed *here* rather than at import. Uvicorn installs
    its own handlers in ``Server.serve()`` before it imports the application, so a
    handler registered at module scope replaces uvicorn's and the process then never
    exits on SIGTERM — it just 503s every request until the orchestrator SIGKILLs it.
    Registering during lifespan means uvicorn is already set up, and we chain to its
    handler so shutdown still happens.
    """
    global _event_executor

    init_db()
    _event_executor = ThreadPoolExecutor(
        max_workers=EVENT_WRITER_WORKERS, thread_name_prefix="event-writer"
    )

    previous_handler = None
    try:
        previous_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _chain_sigterm(previous_handler))
    except ValueError:
        # Not running in the main thread (e.g. TestClient) — no signals to handle.
        pass

    try:
        yield
    finally:
        if _event_executor is not None:
            _event_executor.shutdown(wait=True)
            _event_executor = None


def _chain_sigterm(previous_handler):
    def _handler(signum, frame):
        _handle_sigterm(signum, frame)
        if callable(previous_handler):
            previous_handler(signum, frame)

    return _handler


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _submit_event_write(fn, *args, **kwargs) -> None:
    """
    Queue a durable event write without blocking the response.

    Failures are logged and counted rather than discarded: the previous
    fire-and-forget call dropped the future, so a dead event store returned HTTP 200
    with no log line and no metric, and monitoring silently read an empty table.
    """
    global _event_backlog

    executor = _event_executor
    if executor is None:  # outside the app lifespan (e.g. direct unit tests)
        return

    with _event_backlog_lock:
        if _event_backlog >= EVENT_WRITER_QUEUE_LIMIT:
            EVENT_WRITE_DROPPED_TOTAL.inc()
            logger.error(
                "Event writer saturated (%d queued) — dropping prediction event",
                _event_backlog,
            )
            return
        _event_backlog += 1

    def _done(fut) -> None:
        global _event_backlog
        with _event_backlog_lock:
            _event_backlog -= 1
        exc = fut.exception()
        if exc is not None:
            EVENT_WRITE_FAILURES_TOTAL.inc()
            logger.error("Prediction event write failed: %s", exc, exc_info=exc)

    executor.submit(fn, *args, **kwargs).add_done_callback(_done)


@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    """Reject new requests once SIGTERM has been received."""
    if _shutting_down and request.url.path not in ("/health", "/metrics"):
        return JSONResponse(
            status_code=503,
            content={"error_code": "shutting_down", "message": "Server is shutting down"},
        )
    return await call_next(request)


# Largest request body we will read at all. The per-endpoint MAX_BATCH_SIZE check
# cannot protect us on its own: `payloads: List[RequestModel]` is a FastAPI
# parameter, so Starlette buffers the whole body and Pydantic builds every model
# instance during dependency resolution — the size check runs only after that
# memory has already been committed.
MAX_BODY_BYTES = int(os.environ.get("CHURN_MAX_BODY_BYTES", str(8 * 1024 * 1024)))


@app.middleware("http")
async def body_size_middleware(request: Request, call_next):
    """Reject oversized payloads before the body is read into memory."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError:
            declared = 0
        if declared > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content=ErrorBody(
                    error_code="payload_too_large",
                    message=f"Request body exceeds {MAX_BODY_BYTES} bytes",
                    detail=None,
                ).model_dump(),
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Auth + rate limit helpers
# ---------------------------------------------------------------------------
def _rate_limit() -> str:
    return str(config.get("api", {}).get("rate_limit", "60/minute"))


def _auth_enabled() -> bool:
    return bool(os.environ.get("CHURN_API_KEY", "").strip())


# Refuse to start unauthenticated unless that was an explicit, deliberate choice.
# Without this the service fails OPEN: an unset CHURN_API_KEY silently turns every
# `Depends(verify_api_key)` into a no-op, and `docker compose up` with no .env is
# exactly that case.
ALLOW_ANONYMOUS = _env_flag("CHURN_ALLOW_ANONYMOUS")

if not _auth_enabled() and not ALLOW_ANONYMOUS:
    raise RuntimeError(
        "CHURN_API_KEY is not set, so every authenticated endpoint would be open. "
        "Set CHURN_API_KEY=<secret> to enable authentication, or set "
        "CHURN_ALLOW_ANONYMOUS=1 to explicitly run without it (local development only)."
    )

if not _auth_enabled():
    logger.warning(
        "Running with authentication DISABLED (CHURN_ALLOW_ANONYMOUS=1) — "
        "do not use this configuration outside local development."
    )


def _check_key(expected: str, provided: str | None, hint: str) -> None:
    """Constant-time API key comparison; raises 401 on mismatch."""
    if not provided or not secrets.compare_digest(provided.strip(), expected):
        raise HTTPException(
            status_code=401,
            detail=ErrorBody(
                error_code="unauthorized",
                message="Invalid or missing API key",
                detail=hint,
            ).model_dump(),
        )


def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    if not _auth_enabled():
        return
    _check_key(
        os.environ["CHURN_API_KEY"].strip(),
        x_api_key,
        "Send header X-API-Key matching server CHURN_API_KEY",
    )


def verify_admin_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    """
    Guard administrative endpoints.

    Prefers a dedicated CHURN_ADMIN_API_KEY so a leaked prediction key cannot force
    model reloads; falls back to the regular key when no admin key is configured.
    """
    admin_key = os.environ.get("CHURN_ADMIN_API_KEY", "").strip()
    if not admin_key:
        verify_api_key(x_api_key)
        return
    _check_key(
        admin_key,
        x_api_key,
        "Send header X-API-Key matching server CHURN_ADMIN_API_KEY",
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Log the full detail server-side; the response says what is wrong without
    # echoing the caller's submitted values back at them.
    logger.info("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content=ErrorBody(
            error_code="validation_error",
            message="Request body validation failed",
            detail=_validation_summary(exc),
        ).model_dump(),
    )


def _validation_summary(exc: RequestValidationError) -> str:
    """Field-level summary of a validation failure, without echoing input values."""
    parts = []
    for error in exc.errors()[:20]:
        location = ".".join(str(p) for p in error.get("loc", ()) if p != "body")
        parts.append(f"{location or 'body'}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Give every error response the same shape.

    ``HTTPException(detail=ErrorBody(...))`` is rendered by FastAPI's default handler
    as ``{"detail": {...}}``, while the middleware and validation handlers return the
    body at the top level — so a client parsing ``body["error_code"]`` worked for some
    statuses and broke for others.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "error_code" in detail:
        content = detail
    else:
        content = ErrorBody(
            error_code="http_error",
            message=str(detail) if detail else "Request failed",
            detail=None,
        ).model_dump()
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


RequestModel = generate_request_model()

MAX_BATCH_SIZE = int(os.environ.get("CHURN_MAX_BATCH_SIZE", "100"))

# Batch chunk size for concurrent processing
BATCH_CHUNK_SIZE = int(os.environ.get("CHURN_BATCH_CHUNK_SIZE", "25"))

# Hard ceiling on rows returned by /explain/global regardless of ?limit=
MAX_GLOBAL_IMPORTANCE_FEATURES = int(
    os.environ.get("CHURN_MAX_GLOBAL_IMPORTANCE_FEATURES", "200")
)


def _get_model():
    """Get the model from the thread-safe ModelRegistry."""
    return ModelRegistry.instance().get_model()


THRESHOLD = config["inference"]["threshold"]


# ---------------------------------------------------------------------------
# Health / metrics endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Churn model is running"}


@app.get("/health")
async def health():
    """
    Liveness probe — the process is up and serving.

    Deliberately shallow and always 200 while the process runs; use /ready for
    traffic admission.
    """
    model_info = ModelRegistry.instance().get_info()
    return {
        "status": "ok",
        "model_loaded": model_info["is_loaded"],
        "model_version": model_info["model_version"],
    }


@app.get("/ready")
async def ready(response: Response):
    """
    Readiness probe — can this instance actually serve a prediction?

    ``/health`` previously returned 200 even with a missing or corrupt model.pkl,
    so an orchestrator would route traffic to an instance that 500s every request.
    This performs a real one-row inference against the loaded pipeline.
    """
    try:
        await asyncio.to_thread(_readiness_probe)
    except Exception as exc:
        logger.warning("Readiness probe failed: %s", exc)
        response.status_code = 503
        return ErrorBody(
            error_code="not_ready",
            message="Model is not able to serve predictions",
            detail=str(exc),
        ).model_dump()

    info = ModelRegistry.instance().get_info()
    return {
        "status": "ready",
        "model_version": info["model_version"],
    }


def _readiness_probe() -> None:
    """Run one prediction through the real pipeline to prove the model works."""
    schema = load_feature_schema()
    probe_row = {feature: _PROBE_VALUES.get(feature, 0) for feature in schema}
    df = pd.DataFrame([probe_row])
    model = _get_model()
    model.predict_proba(df[schema])


# Neutral placeholder values for the readiness probe. Numeric zeros work for
# scaled columns, and unseen categories are tolerated because the pipeline's
# OneHotEncoder is configured with handle_unknown="ignore".
_PROBE_VALUES: dict[str, object] = {}


@app.get("/metrics")
async def metrics():
    body, content_type = render_latest()
    return Response(content=body, media_type=content_type)


# ---------------------------------------------------------------------------
# Model management endpoint (for hot-reload)
# ---------------------------------------------------------------------------
@app.post("/admin/reload-model")
@limiter.limit("5/minute")
async def reload_model(request: Request, _: None = Depends(verify_admin_key)):
    """
    Hot-reload the production model without server restart.

    The ModelRegistry acquires an exclusive write lock, ensuring all
    in-flight predictions complete before the swap. New predictions
    block briefly until the new model is loaded.
    """
    await asyncio.to_thread(ModelRegistry.instance().reload)
    info = ModelRegistry.instance().get_info()
    logger.info("Model hot-reloaded via admin endpoint | version=%s", info["model_version"])
    return {"status": "reloaded", "model_info": info}


# ---------------------------------------------------------------------------
# Synchronous inference helper (runs in thread pool)
# ---------------------------------------------------------------------------
def _run_single_inference(row: dict) -> dict:
    """
    CPU-bound inference for a single row.

    This function runs in a thread pool worker via asyncio.to_thread().
    The ModelRegistry.get_model() call is thread-safe (ReadWriteLock).
    """
    df = pd.DataFrame([row])
    df = build_features(df, training=False)
    df_valid = validate_inference_data(df)
    model = _get_model()
    prob = float(model.predict_proba(df_valid)[:, 1][0])
    return {"probability": prob, "prediction": int(prob >= THRESHOLD)}


def _run_batch_inference(rows: list[dict]) -> list[float]:
    """
    CPU-bound inference for a batch of rows.

    Processes the entire chunk as a single DataFrame for vectorized
    computation — much faster than per-row prediction.
    """
    df = pd.DataFrame(rows)
    df = build_features(df, training=False)
    df_valid = validate_inference_data(df)
    model = _get_model()
    return model.predict_proba(df_valid)[:, 1].tolist()


# ---------------------------------------------------------------------------
# Single-row predict (async)
# ---------------------------------------------------------------------------
@app.post("/predict")
@limiter.limit(_rate_limit())
async def predict(
    request: Request,
    payload: RequestModel,
    x_subject_id: str | None = Header(None, alias="X-Subject-Id"),
    _: None = Depends(verify_api_key),
):
    """
    Accepts raw feature row and returns churn probability.

    The CPU-bound model inference is offloaded to a thread via
    asyncio.to_thread() so the event loop remains non-blocking.
    """
    request_id = uuid.uuid4().hex
    start_time = time.time()
    logger.info("Received prediction request | request_id=%s", request_id)

    try:
        row = payload.model_dump()
        # Offload CPU-bound inference to thread pool
        result = await asyncio.to_thread(_run_single_inference, row)
    except ValueError as e:
        logger.warning("Validation failed | request_id=%s | %s", request_id, e)
        REQUESTS_TOTAL.labels(path="/predict", method="POST", status="400").inc()
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(
                error_code="invalid_input",
                message="Input validation failed",
                detail=str(e),
            ).model_dump(),
        ) from e
    except Exception as e:
        logger.exception("Prediction failed | request_id=%s", request_id)
        INFERENCE_ERRORS_TOTAL.inc()
        REQUESTS_TOTAL.labels(path="/predict", method="POST", status="500").inc()
        raise HTTPException(
            status_code=500,
            detail=ErrorBody(
                error_code="inference_error",
                message="Model inference failed",
                detail=None,
            ).model_dump(),
        ) from e

    prob = result["probability"]
    prediction = result["prediction"]
    latency = time.time() - start_time

    # Persist asynchronously on the dedicated event-writer pool.
    _submit_event_write(
        store_prediction_event,
        request_id=request_id,
        raw_features=row,
        probability=prob,
        prediction=prediction,
        latency_seconds=latency,
        subject_id=x_subject_id,
    )

    REQUEST_LATENCY_SECONDS.labels(path="/predict", method="POST").observe(latency)
    REQUESTS_TOTAL.labels(path="/predict", method="POST", status="200").inc()
    logger.info(
        "Prediction | request_id=%s | prob=%.4f | pred=%s | latency=%.4fs",
        request_id,
        prob,
        prediction,
        latency,
    )

    return {
        "request_id": request_id,
        "churn_probability": round(prob, 4),
        "prediction": prediction,
        "threshold": THRESHOLD,
        "latency_seconds": round(latency, 4),
    }


# ---------------------------------------------------------------------------
# Batch predict with CONCURRENT chunk processing (async)
# ---------------------------------------------------------------------------
@app.post("/predict/batch")
@limiter.limit(_rate_limit())
async def predict_batch(
    request: Request,
    payloads: list[RequestModel],
    _: None = Depends(verify_api_key),
):
    """
    Accepts a list of feature rows and returns churn probabilities.

    Concurrency strategy:
    - Splits the batch into chunks of BATCH_CHUNK_SIZE
    - Each chunk runs inference in a separate thread via asyncio.to_thread()
    - asyncio.gather() runs all chunks concurrently
    - Results are reassembled in order

    This achieves parallelism: while one chunk is waiting on GIL release
    during sklearn's C-extension predict, other chunks can proceed with
    Python-level DataFrame construction.
    """
    if len(payloads) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(
                error_code="batch_too_large",
                message=f"Batch size {len(payloads)} exceeds maximum of {MAX_BATCH_SIZE}",
                detail=None,
            ).model_dump(),
        )

    batch_id = uuid.uuid4().hex
    start_time = time.time()
    logger.info("Batch request | batch_id=%s | size=%d", batch_id, len(payloads))

    rows = [p.model_dump() for p in payloads]

    # Split into chunks for concurrent processing
    chunks = [rows[i : i + BATCH_CHUNK_SIZE] for i in range(0, len(rows), BATCH_CHUNK_SIZE)]

    try:
        # Process all chunks concurrently using asyncio.gather
        chunk_results = await asyncio.gather(
            *[asyncio.to_thread(_run_batch_inference, chunk) for chunk in chunks]
        )
    except ValueError as e:
        REQUESTS_TOTAL.labels(path="/predict/batch", method="POST", status="400").inc()
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(
                error_code="invalid_input",
                message="Batch validation failed",
                detail=str(e),
            ).model_dump(),
        ) from e
    except Exception as e:
        INFERENCE_ERRORS_TOTAL.inc()
        REQUESTS_TOTAL.labels(path="/predict/batch", method="POST", status="500").inc()
        raise HTTPException(
            status_code=500,
            detail=ErrorBody(
                error_code="inference_error",
                message="Batch inference failed",
                detail=None,
            ).model_dump(),
        ) from e

    # Flatten chunk results back into a single list
    all_probs = []
    for chunk_probs in chunk_results:
        all_probs.extend(chunk_probs)

    latency = time.time() - start_time
    REQUEST_LATENCY_SECONDS.labels(path="/predict/batch", method="POST").observe(latency)
    REQUESTS_TOTAL.labels(path="/predict/batch", method="POST", status="200").inc()

    results = []
    per_row_latency = latency / max(len(all_probs), 1)
    for i, prob in enumerate(all_probs):
        p = float(prob)
        prediction = int(p >= THRESHOLD)
        results.append({
            "index": i,
            "request_id": f"{batch_id}-{i}",
            "churn_probability": round(p, 4),
            "prediction": prediction,
        })
        # Batch predictions were previously never persisted, so every row served
        # this way was invisible to the event store, drift detection and the
        # outbox — monitoring silently saw only single-prediction traffic.
        _submit_event_write(
            store_prediction_event,
            request_id=f"{batch_id}-{i}",
            raw_features=rows[i],
            probability=p,
            prediction=prediction,
            latency_seconds=per_row_latency,
        )

    logger.info(
        "Batch complete | batch_id=%s | count=%d | chunks=%d | latency=%.4fs",
        batch_id,
        len(results),
        len(chunks),
        latency,
    )

    return {
        "batch_id": batch_id,
        "count": len(results),
        "threshold": THRESHOLD,
        "latency_seconds": round(latency, 4),
        "predictions": results,
    }


@app.get("/predict")
async def predict_get_help():
    return {
        "message": "Use POST /predict with JSON body.",
        "hint": "Open /docs for the interactive request form.",
    }


# ---------------------------------------------------------------------------
# Ground-truth feedback
# ---------------------------------------------------------------------------
class FeedbackBody(BaseModel):
    """Observed outcome for a past prediction."""

    model_config = ConfigDict(extra="forbid")

    label: int = Field(..., ge=0, le=1, description="1 if the customer churned, else 0")


@app.post("/feedback/{request_id}")
@limiter.limit(_rate_limit())
async def submit_feedback(
    request: Request,
    request_id: str,
    body: FeedbackBody,
    _: None = Depends(verify_api_key),
):
    """
    Attach the observed outcome to a prediction this service made.

    This is the input the system needs to measure whether the model is still
    *correct*. Without it, monitoring can only compare input distributions and will
    never detect performance decay — a model can rot while every drift metric stays
    green.
    """
    updated = await asyncio.to_thread(record_label, request_id, body.label)

    LABELS_RECORDED_TOTAL.labels(outcome="matched" if updated else "unknown").inc()

    if not updated:
        raise HTTPException(
            status_code=404,
            detail=ErrorBody(
                error_code="unknown_request_id",
                message="No prediction found for that request_id",
                detail=None,
            ).model_dump(),
        )

    logger.info("Label recorded | request_id=%s | label=%s", request_id, body.label)
    return {"request_id": request_id, "label": body.label, "status": "recorded"}


@app.delete("/subject/{subject_id}")
@limiter.limit("30/minute")
async def erase_subject(
    request: Request,
    subject_id: str,
    _: None = Depends(verify_admin_key),
):
    """
    Erase every stored prediction for a customer (GDPR right to erasure).

    Matching happens on the salted pseudonymous key derived from the identifier, so
    the event store never has to hold the identifier itself.
    """
    from churn_system.events.predictions import purge_subject

    deleted = await asyncio.to_thread(purge_subject, subject_id)
    logger.info("Subject erasure | deleted_rows=%d", deleted)
    return {"deleted": deleted, "status": "erased"}


# ---------------------------------------------------------------------------
# Explainable AI endpoint
# ---------------------------------------------------------------------------
@app.post("/explain")
@limiter.limit(_rate_limit())
async def explain(
    request: Request,
    payload: RequestModel,
    _: None = Depends(verify_api_key),
):
    """
    Return SHAP-based feature explanations for a single prediction.

    Explains WHY the model predicted a specific churn probability by
    returning per-feature SHAP contributions, top positive drivers
    (pushing toward churn), and top negative drivers (pushing away).
    """
    from churn_system.explainability.shap_explainer import explain_prediction

    request_id = uuid.uuid4().hex
    start_time = time.time()
    EXPLANATION_REQUESTS_TOTAL.inc()

    logger.info("Explanation request | request_id=%s", request_id)

    try:
        row = payload.model_dump()
        result = await asyncio.to_thread(explain_prediction, row)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=ErrorBody(
                error_code="explainer_not_ready",
                message="Training reference data not available",
                detail=str(e),
            ).model_dump(),
        ) from e
    except Exception as e:
        logger.exception("Explanation failed | request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=ErrorBody(
                error_code="explanation_error",
                message="Failed to generate explanation",
                detail=None,
            ).model_dump(),
        ) from e

    latency = time.time() - start_time
    EXPLANATION_LATENCY_SECONDS.observe(latency)

    logger.info(
        "Explanation complete | request_id=%s | latency=%.4fs",
        request_id,
        latency,
    )

    return {
        "request_id": request_id,
        **result,
        "latency_seconds": round(latency, 4),
    }


# ---------------------------------------------------------------------------
# Global feature importance endpoint
# ---------------------------------------------------------------------------
@app.get("/explain/global")
@limiter.limit(_rate_limit())
async def global_importance(
    request: Request,
    limit: int = 50,
    _: None = Depends(verify_api_key),
):
    """
    Return global SHAP feature importance (mean |SHAP| across background data).

    Truncated to the top ``limit`` features by default. The untruncated response is
    one row per *encoded* feature, so with one-hot encoded geography it enumerates
    every city, ZIP code and lat/long pair present in the training data — a full
    disclosure of the customer base's geographic footprint.
    """
    from churn_system.explainability.shap_explainer import compute_global_importance

    try:
        result = await asyncio.to_thread(compute_global_importance)
    except Exception as e:
        logger.exception("Global importance computation failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorBody(
                error_code="global_importance_error",
                message="Failed to compute global feature importance",
                detail=None,
            ).model_dump(),
        ) from e

    importance = result.get("feature_importance", [])
    capped = max(1, min(int(limit), MAX_GLOBAL_IMPORTANCE_FEATURES))

    return {
        **result,
        "feature_importance": importance[:capped],
        "total_features": len(importance),
        "truncated": len(importance) > capped,
    }


# ---------------------------------------------------------------------------
# Monitoring dashboard endpoint
# ---------------------------------------------------------------------------
@app.get("/monitoring/dashboard")
async def monitoring_dashboard(_: None = Depends(verify_api_key)):
    """
    Returns a consolidated view of all monitoring metrics.

    Aggregates data quality reports, calibration reports, drift status,
    and model health into a single JSON response for dashboarding.
    """
    import json
    from pathlib import Path

    monitoring_dir = Path(CONFIG["paths"]["monitoring_dir"])
    dashboard = {"available_reports": []}

    report_files = {
        "data_quality": monitoring_dir / "data_quality_report.json",
        "calibration": monitoring_dir / "calibration_report.json",
        "prediction": monitoring_dir / "prediction_report.json",
        "health": monitoring_dir / "health_report.json",
    }

    for name, path in report_files.items():
        if path.exists():
            with open(path) as f:
                dashboard[name] = json.load(f)
            dashboard["available_reports"].append(name)
        else:
            dashboard[name] = None

    # Model registry info, minus the on-disk path (server filesystem layout is not
    # something an API response should disclose).
    registry_info = ModelRegistry.instance().get_info()
    dashboard["model_info"] = {
        k: v for k, v in registry_info.items() if k != "model_path"
    }

    return dashboard
