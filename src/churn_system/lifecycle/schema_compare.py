import json
from pathlib import Path

def load_schema(metadata_path : Path):
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    return set(meta["feature_schema"])

def compare_feature_schemas(prod_meta: Path, challenger_meta: Path):
    """
    Compare production and challenger feature schemas.
    """
    
    prod_schema = load_schema(prod_meta)
    challenger_schema = load_schema(challenger_meta)
    
    added = challenger_schema - prod_schema
    removed = prod_schema - challenger_schema
    
    return {
        "added_features" : list(added),
        "removed_features" : list(removed),
        "is_identical" : len(added) == 0 and len(removed) == 0
    }