from importlib import metadata
from pathlib import Path
import json

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["api"])

_METADATA_CACHE = None


def load_model_contract():
    """
    Load Production model metadata once and cache it.
    """
    
    global _METADATA_CACHE

    if _METADATA_CACHE is not None:
        return _METADATA_CACHE
    
    metadata_path = Path(CONFIG["paths"]["production_model"]).parent / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Production metadata not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        _METADATA_CACHE = json.load(f)
        
    logger.info("Model Contract loaded into memory.")
    
    return _METADATA_CACHE

def get_feature_schema():
    """
    Return feature schema expected by deployed model.
    """
    metadata = load_model_contract()
    return metadata["feature_schema"]