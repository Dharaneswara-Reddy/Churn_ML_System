"""
Churn prediction HTTP API.

- Typed request body (Pydantic, no Any)
- Optional API key (CHURN_API_KEY)
- Rate limiting (slowapi)
- Structured error responses
- SIGTERM graceful shutdown
- Batch prediction endpoint
"""

from __future__ import annotations

import os
import pickle
import signal
import time
import uuid
from functools import lru_cache
from pathlib import Path
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


@lru_cache(maxsize=1)
def get_model():
    """Load production model with LRU cache (loaded once, reused)."""
    model_path = Path(CONFIG["paths"]["production_model"])
    with open(model_path, "rb") as f:
        return pickle.load(f)


THRESHOLD = config["inference"]["threshold"]


# ---------------------------------------------------------------------------
# Health / metrics endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Churn model is running"}


@app.get("/health")
def health():
    """Readiness / liveness probe endpoint."""
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    body, content_type = render_latest()
    return Response(content=body, media_type=content_type)


# ---------------------------------------------------------------------------
# Single-row predict
# ---------------------------------------------------------------------------
@app.post("/predict")
@limiter.limit(_rate_limit())
def predict(
    request: Request,
    payload: RequestModel,
    _: None = Depends(verify_api_key),
):
    """
    Accepts raw feature row and returns churn probability.
    """
    request_id = uuid.uuid4().hex
    start_time = time.time()
    logger.info("Received prediction request | request_id=%s", request_id)

    try:
        row = payload.model_dump()
        df = pd.DataFrame([row])
        df = build_features(df, training=False)
        df_valid = validate_inference_data(df)
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
        logger.exception("Unexpected validation error | request_id=%s", request_id)
        REQUESTS_TOTAL.labels(path="/predict", method="POST", status="400").inc()
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(
                error_code="invalid_input",
                message="Input validation failed",
                detail=str(e),
            ).model_dump(),
        ) from e

    try:
        prob = float(get_model().predict_proba(df_valid)[:, 1][0])
        prediction = int(prob >= THRESHOLD)
        latency = time.time() - start_time
        store_prediction_event(
            request_id=request_id,
            raw_features=row,
            probability=prob,
            prediction=prediction,
            latency_seconds=latency,
        )
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

    latency = time.time() - start_time
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
# Batch predict (configurable max via CHURN_MAX_BATCH_SIZE, default 100)
# ---------------------------------------------------------------------------
@app.post("/predict/batch")
@limiter.limit(_rate_limit())
def predict_batch(
    request: Request,
    payloads: List[RequestModel],
    _: None = Depends(verify_api_key),
):
    """
    Accepts a list of feature rows and returns churn probabilities.

    Maximum batch size is controlled by the ``CHURN_MAX_BATCH_SIZE``
    environment variable (default: 100).
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

    try:
        rows = [p.model_dump() for p in payloads]
        df = pd.DataFrame(rows)
        df = build_features(df, training=False)
        df_valid = validate_inference_data(df)
    except (ValueError, Exception) as e:
        REQUESTS_TOTAL.labels(path="/predict/batch", method="POST", status="400").inc()
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(
                error_code="invalid_input",
                message="Batch validation failed",
                detail=str(e),
            ).model_dump(),
        ) from e

    try:
        model = get_model()
        probs = model.predict_proba(df_valid)[:, 1]
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

    latency = time.time() - start_time
    REQUEST_LATENCY_SECONDS.labels(path="/predict/batch", method="POST").observe(latency)
    REQUESTS_TOTAL.labels(path="/predict/batch", method="POST", status="200").inc()

    results = []
    for i, prob in enumerate(probs):
        p = float(prob)
        results.append({
            "index": i,
            "churn_probability": round(p, 4),
            "prediction": int(p >= THRESHOLD),
        })

    return {
        "batch_id": batch_id,
        "count": len(results),
        "threshold": THRESHOLD,
        "latency_seconds": round(latency, 4),
        "predictions": results,
    }


@app.get("/predict")
def predict_get_help():
    return {
        "message": "Use POST /predict with JSON body.",
        "hint": "Open /docs for the interactive request form.",
    }
