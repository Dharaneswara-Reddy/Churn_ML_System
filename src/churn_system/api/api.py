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
import signal
import time
import uuid
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from churn_system.api.errors import ErrorBody
from churn_system.api.schema_generator import generate_request_model
from churn_system.config.config import CONFIG, load_config
from churn_system.events.predictions import store_prediction_event
from churn_system.features.build_features import build_features
from churn_system.logging.logger import get_logger
from churn_system.observability.metrics import (
    INFERENCE_ERRORS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    render_latest,
)
from churn_system.schema import validate_inference_data
from churn_system.serving.model_registry import ModelRegistry

logger = get_logger(__name__, CONFIG["logging"]["api"])
config = load_config()

limiter = Limiter(
    key_func=get_remote_address,
    enabled=not os.environ.get("CHURN_DISABLE_RATE_LIMIT"),
)
app = FastAPI(title="Churn Prediction API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Graceful shutdown on SIGTERM (container orchestrator → drain → exit)
# ---------------------------------------------------------------------------
_shutting_down = False


def _handle_sigterm(signum, frame):  # noqa: ARG001
    global _shutting_down
    _shutting_down = True
    logger.info("SIGTERM received — draining in-flight requests before shutdown")


signal.signal(signal.SIGTERM, _handle_sigterm)


@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    """Reject new requests once SIGTERM has been received."""
    if _shutting_down and request.url.path not in ("/health", "/metrics"):
        return JSONResponse(
            status_code=503,
            content={"error_code": "shutting_down", "message": "Server is shutting down"},
        )
    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# Auth + rate limit helpers
# ---------------------------------------------------------------------------
def _rate_limit() -> str:
    return str(config.get("api", {}).get("rate_limit", "60/minute"))


def _auth_enabled() -> bool:
    return bool(os.environ.get("CHURN_API_KEY", "").strip())


def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    if not _auth_enabled():
        return
    expected = os.environ["CHURN_API_KEY"].strip()
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(
            status_code=401,
            detail=ErrorBody(
                error_code="unauthorized",
                message="Invalid or missing API key",
                detail="Send header X-API-Key matching server CHURN_API_KEY",
            ).model_dump(),
        )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ErrorBody(
            error_code="validation_error",
            message="Request body validation failed",
            detail=str(exc.errors()),
        ).model_dump(),
    )


RequestModel = generate_request_model()

MAX_BATCH_SIZE = int(os.environ.get("CHURN_MAX_BATCH_SIZE", "100"))

# Batch chunk size for concurrent processing
BATCH_CHUNK_SIZE = int(os.environ.get("CHURN_BATCH_CHUNK_SIZE", "25"))


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
    """Readiness / liveness probe endpoint."""
    model_info = ModelRegistry.instance().get_info()
    return {
        "status": "ok",
        "model_loaded": model_info["is_loaded"],
        "model_version": model_info["model_version"],
    }


@app.get("/metrics")
async def metrics():
    body, content_type = render_latest()
    return Response(content=body, media_type=content_type)


# ---------------------------------------------------------------------------
# Model management endpoint (for hot-reload)
# ---------------------------------------------------------------------------
@app.post("/admin/reload-model")
async def reload_model(_: None = Depends(verify_api_key)):
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

    # Store prediction event (fire-and-forget in thread pool)
    asyncio.get_event_loop().run_in_executor(
        None,
        lambda: store_prediction_event(
            request_id=request_id,
            raw_features=row,
            probability=prob,
            prediction=prediction,
            latency_seconds=latency,
        ),
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
    payloads: List[RequestModel],
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
    for i, prob in enumerate(all_probs):
        p = float(prob)
        results.append({
            "index": i,
            "churn_probability": round(p, 4),
            "prediction": int(p >= THRESHOLD),
        })

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
