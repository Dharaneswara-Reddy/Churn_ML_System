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
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from churn_system.api.errors import ErrorBody
from churn_system.api.schema_generator import (
    generate_request_model,
    load_feature_schema,
    load_feature_types,
)
from churn_system.config.config import CONFIG, load_config
from churn_system.events.db import init_db
from churn_system.events.predictions import record_label, store_prediction_event
from churn_system.features.build_features import build_features
from churn_system.inference.model_contract import load_model_contract
from churn_system.logging.logger import get_logger
from churn_system.observability.metrics import (
    DEPRECATED_REQUEST_FIELDS_TOTAL,
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
from churn_system.observability.state_metrics import refresh_state_metrics
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


BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})


@app.middleware("http")
async def body_size_middleware(request: Request, call_next):
    """Reject oversized payloads before the body is read into memory."""
    content_length = request.headers.get("content-length")

    # A chunked request declares no Content-Length, so a size check that trusts
    # that header alone is trivially bypassed — a 42MB body was accepted and
    # fully materialised in memory despite an 8MB cap. Requiring a declared
    # length on body-bearing methods closes that hole; this API only ever
    # receives JSON of known size.
    if content_length is None and request.method in BODY_METHODS:
        return JSONResponse(
            status_code=411,
            content=ErrorBody(
                error_code="length_required",
                message="Content-Length header is required",
                detail="Chunked transfer encoding is not accepted on this endpoint",
            ).model_dump(),
        )

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


def _build_request_model():
    """
    Build the dynamic request model, degrading rather than killing the process.

    The request schema is derived from the production bundle's metadata, so a
    missing or unreadable bundle used to raise here — at module import, before
    ``lifespan`` ran and before any route was bound. The process therefore died
    with a bare traceback instead of serving the structured 503 that ``/ready``
    exists to return: under an orchestrator that is CrashLoopBackOff with no
    readiness signal and no way to tell a missing model from a crashed app.

    Falling back to a permissive model keeps the process bootable and observable.
    Serving is NOT silently degraded: ``validate_inference_data`` still enforces
    the real schema on every request, and ``/ready`` reports 503 until a valid
    bundle is present, so an instance in this state never passes a readiness gate.
    """
    try:
        return generate_request_model()
    except Exception:
        logger.exception(
            "Could not build the request schema from the production bundle. The API "
            "will start so /health and /ready are reachable, but /ready will report "
            "503 and predictions will fail until a valid bundle is promoted."
        )

        class UnconfiguredPredictionRequest(BaseModel):
            model_config = ConfigDict(extra="allow")

        return UnconfiguredPredictionRequest


RequestModel = _build_request_model()


def _payload_to_row(payload) -> dict:
    """
    Turn a validated request into a feature row, dropping deprecated fields.

    The request model accepts fields the current champion no longer uses so that
    clients written against an older schema keep working (see
    ``api/schema_generator``). Those fields must not travel any further: they
    would be carried into ``build_features`` and, more importantly, into the
    redaction-and-storage path, where the whole point of removing geography was
    that it stops being collected at all.

    Usage is counted per field. Without that, "can we retire the shim?" has no
    answer except guessing.
    """
    row = payload.model_dump()

    for field in _deprecated_fields():
        if row.pop(field, None) is not None:
            DEPRECATED_REQUEST_FIELDS_TOTAL.labels(field=field).inc()

    return row


def _deprecated_fields() -> tuple[str, ...]:
    """
    Deprecated field names for the *current* request model.

    Recomputed rather than cached because ``POST /admin/reload-model`` can swap
    the champion — and therefore the deprecated set — without a restart.
    """
    try:
        from churn_system.api.schema_generator import deprecated_request_fields

        return tuple(deprecated_request_fields())
    except Exception:
        return ()

MAX_BATCH_SIZE = int(os.environ.get("CHURN_MAX_BATCH_SIZE", "100"))

# How many rows go into one inference call. Defaults to the whole batch — i.e. no
# chunking.
#
# Chunking was introduced to "parallelise" batch inference by fanning chunks out
# through asyncio.gather + asyncio.to_thread. Measured, it does the opposite:
# scikit-learn's predict_proba holds the GIL for the whole traversal, so extra
# threads add no parallelism while multiplying the fixed per-call cost
# (DataFrame construction, schema validation, reindexing) by the number of chunks.
#
# Measured on a 100-row batch, one uvicorn worker, 25 reps with outliers
# trimmed, across two independent runs:
#
#     chunk= 10 -> 266-279 ms   358-376 rows/s
#     chunk= 25 -> 128-129 ms   775-780 rows/s   <- the previous default
#     chunk= 50 ->  77- 88 ms  1136-1299 rows/s
#     chunk=100 ->  54- 55 ms  1819-1847 rows/s  <- one chunk, the new default
#
# Monotonic in both runs: every split costs throughput. Run-to-run variance is
# around 10% and never reorders the rows. The knob is kept because it bounds peak
# memory for a single inference call, which matters if MAX_BATCH_SIZE is raised
# well beyond 100 — but it is a memory/fairness control, not a throughput one.
BATCH_CHUNK_SIZE = int(os.environ.get("CHURN_BATCH_CHUNK_SIZE", str(MAX_BATCH_SIZE)))

# Hard ceiling on rows returned by /explain/global regardless of ?limit=
MAX_GLOBAL_IMPORTANCE_FEATURES = int(
    os.environ.get("CHURN_MAX_GLOBAL_IMPORTANCE_FEATURES", "200")
)


def _get_model():
    """Get the model from the thread-safe ModelRegistry."""
    return ModelRegistry.instance().get_model()


def operating_threshold() -> float:
    """
    The decision threshold for the model currently in production.

    Read from the model bundle's metadata when present, so a model tuned at 0.28
    is never served at someone else's 0.5. Falls back to the configured default
    for bundles trained before threshold selection existed, which keeps older
    bundles servable rather than failing closed on a missing key.
    """
    try:
        contract = load_model_contract()
    except Exception:  # bundle unreadable — the caller will fail on the model anyway
        return float(config["inference"]["threshold"])

    value = contract.get("operating_threshold")
    if value is None:
        return float(config["inference"]["threshold"])
    return float(value)


# Backwards-compatible module attribute. Tests and older callers read
# `api.THRESHOLD`; it now reflects the bundle loaded at import.
THRESHOLD = operating_threshold()


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


def _probe_row() -> dict[str, object]:
    """
    One realistic feature row for the readiness probe.

    Taken from the training reference when available, so the probe exercises
    values the fitted encoders actually saw. Filling every field with 0 does not
    work: the pipeline one-hot encodes string columns, and handing an integer to
    a fitted OneHotEncoder whose categories are strings makes sklearn call
    np.isnan() on a mixed-type array, which raises TypeError — so the probe
    failed 100% of the time regardless of model health.
    """
    schema = load_feature_schema()
    reference = Path(CONFIG["paths"]["training_reference"])

    types = load_feature_types()

    if reference.exists():
        frame = pd.read_csv(reference, nrows=1)
        if not frame.empty and all(column in frame.columns for column in schema):
            row = frame[schema].iloc[0].to_dict()
            # The reference is a CSV, so dtypes are re-inferred on read: a
            # numeric-looking categorical such as "Zip Code" comes back as int64
            # while the fitted encoder holds string categories. Restore the types
            # the bundle declares before scoring.
            return {
                feature: (str(value) if types.get(feature) == "str" else value)
                for feature, value in row.items()
            }

    # No reference on disk — fall back to type-appropriate neutral values rather
    # than zeros, so categorical columns stay strings.
    return {
        feature: (0 if types.get(feature) in {"int", "float", "bool"} else "unknown")
        for feature in schema
    }


def _readiness_probe() -> None:
    """
    Score one row through the real serving path to prove the model works.

    Deliberately uses build_features + validate_inference_data — the same
    functions a real request goes through — so the probe cannot pass while the
    actual prediction path is broken.
    """
    frame = pd.DataFrame([_probe_row()])
    frame = build_features(frame, training=False)
    validated = validate_inference_data(frame)
    model = _get_model()
    model.predict_proba(validated)


@app.get("/metrics")
@limiter.limit("120/minute")
async def metrics(request: Request):
    """
    Prometheus scrape endpoint.

    Refreshes gauges derived from persisted lifecycle state before rendering.
    Drift, champion version and outbox backlog are produced by the scheduler and
    worker processes, whose in-memory registries Prometheus never scrapes — so
    without this the drift alert could not fire at all.
    """
    await asyncio.to_thread(refresh_state_metrics)
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
    threshold = operating_threshold()
    return {"probability": prob, "prediction": int(prob >= threshold), "threshold": threshold}


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
        row = _payload_to_row(payload)
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
        "threshold": result["threshold"],
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
    - Splits the batch into chunks of BATCH_CHUNK_SIZE (by default: one chunk).
    - Each chunk runs inference in a worker thread via asyncio.to_thread(), so the
      event loop stays free to accept other requests.
    - Results are reassembled in the caller's original order.

    Note that multiple chunks do **not** buy parallelism. sklearn's predict_proba
    holds the GIL for the traversal, so chunking only multiplies the fixed
    per-call cost — measured at 5x lower throughput at chunk=10 than unchunked
    (see BATCH_CHUNK_SIZE). The single to_thread hop is what keeps the event loop
    responsive; the chunk loop is a memory bound, not a speed-up.
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

    rows = [_payload_to_row(p) for p in payloads]

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

    batch_threshold = operating_threshold()
    results = []
    per_row_latency = latency / max(len(all_probs), 1)
    for i, prob in enumerate(all_probs):
        p = float(prob)
        prediction = int(p >= batch_threshold)
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
        "threshold": batch_threshold,
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
        row = _payload_to_row(payload)
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
