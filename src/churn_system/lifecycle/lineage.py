"""
Model Lineage Tracking. 

Maintains history of all promoted models.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

LINEAGE_PATH = Path("models/lineage/lineage.json")
LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_lineage():
    if LINEAGE_PATH.exists():
        with open(LINEAGE_PATH, "r") as f:
            return json.load(f)
    return []

def save_lineage(data):
    with open(LINEAGE_PATH, "w") as f:
        json.dump(data,f,indent=2)
        
def record_lineage(model_version : str,metrics : dict,dataset_used : str,trigger : str,parent_model: str | None,):
    """
    Append a lineage record for a promoted model.
    """
    
    lineage = load_lineage()
    
    record = {
        "model_version" : model_version,
        "timestamp" : datetime.now(timezone.utc).isoformat(),
        "dataset" : dataset_used,
        "trigger" : trigger,
        "parents_model" : parent_model,
        "metrics" : metrics,
    }
    
    lineage.append(record)
    save_lineage(lineage)
    
    print(f"Lineage recorded for {model_version}")