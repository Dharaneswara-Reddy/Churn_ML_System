from functools import lru_cache

from churn_system.artifacts import production_model_path, validate_model_bundle
from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["api"])


@lru_cache(maxsize=1)
def load_model_contract():
    """
    Load Production model metadata once and cache it.
    """

    metadata = validate_model_bundle(production_model_path())
    logger.info("Model Contract loaded into memory.")
    return metadata


def get_feature_schema():
    """
    Return feature schema expected by deployed model.
    """
    metadata = load_model_contract()
    return metadata["feature_schema"]


def clear_model_contract_cache() -> None:
    load_model_contract.cache_clear()
