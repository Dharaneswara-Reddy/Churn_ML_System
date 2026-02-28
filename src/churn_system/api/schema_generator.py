"""
Automatic API Schema Generator

Builds FastAPI request schema dynamically
from production model metadata.
"""


from pathlib import Path
from pydantic import create_model
from typing import Dict, Any

from churn_system.config.config import CONFIG

def load_feature_schema():
    """
    Load feature schema from production metadata.
    """

    metadata_path = (
        Path(CONFIG["paths"]["production_model"])
        .parent / "metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Production metadata not found."
        )

    import json

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata["feature_schema"]


def generate_request_model():
    """
    Dynamically create Pydantic request model.
    """

    features = load_feature_schema()

    fields: Dict[str, tuple] = {}

    # Default → accept any type (safe starting point)
    for feature in features:
        fields[feature] = (Any, ...)

    RequestModel = create_model(
        "DynamicPredictionRequest",
        **fields
    )

    return RequestModel