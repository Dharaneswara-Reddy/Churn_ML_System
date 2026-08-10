"""
Notify the serving layer that the production model changed.

Promotion only rewrites bytes on disk. A running API process holds an already
unpickled model plus an ``lru_cache``d copy of the feature contract, so without an
explicit reload it keeps serving the previous model — and, worse, keeps ordering
inference columns by the previous feature schema — until the process restarts.
"""

from __future__ import annotations

import os

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])

RELOAD_TIMEOUT_SECONDS = 10


def _reload_endpoints() -> list[str]:
    """Serving instances to notify, as a comma-separated CHURN_RELOAD_ENDPOINTS list."""
    raw = os.environ.get("CHURN_RELOAD_ENDPOINTS", "").strip()
    return [url.strip() for url in raw.split(",") if url.strip()]


def reload_in_process() -> None:
    """
    Refresh the model and contract caches inside *this* process.

    Correct and sufficient when serving and the lifecycle share a process; a no-op
    worth doing regardless, since it also covers single-process deployments.
    """
    from churn_system.inference.model_contract import clear_model_contract_cache
    from churn_system.serving.model_registry import ModelRegistry

    clear_model_contract_cache()
    registry = ModelRegistry.instance()
    if registry.get_info()["is_loaded"]:
        registry.reload()
        logger.info("In-process model registry reloaded.")


def notify_serving_reload() -> None:
    """
    Reload locally, then ask any configured remote serving instances to reload.

    Remote notification is best-effort: a failure is logged loudly but does not fail
    the promotion, because the model on disk is already correct and a restart will
    pick it up.
    """
    reload_in_process()

    endpoints = _reload_endpoints()
    if not endpoints:
        logger.warning(
            "No CHURN_RELOAD_ENDPOINTS configured — remote API instances will keep "
            "serving the previous model until they are restarted."
        )
        return

    import requests

    headers = {}
    admin_key = os.environ.get("CHURN_ADMIN_API_KEY") or os.environ.get("CHURN_API_KEY")
    if admin_key:
        headers["X-API-Key"] = admin_key.strip()

    for url in endpoints:
        try:
            response = requests.post(url, headers=headers, timeout=RELOAD_TIMEOUT_SECONDS)
            response.raise_for_status()
            logger.info("Reload acknowledged by %s", url)
        except Exception:
            logger.exception("Reload notification failed for %s", url)
