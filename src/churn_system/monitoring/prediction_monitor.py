"""
Prediction Monitoring Module

Analyzes live inference predictions 
to understand model behaviour in production 
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["monitoring"])

PRED_PATH = Path("data/inference_logs/predictions.csv")
REPORT_DIR = Path("models/monitoring")
REPORT_DIR.mkdir(parents = True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "prediction_report.json"


def generate_prediction_report():
    """
    Analyze production predictions and generate monitoring metrics.
    """
    
    if not PRED_PATH.exists():
        logger.warning("No prediction logs found.")
        return 
    
    df = pd.read_csv(PRED_PATH)
    
    if df.empty:
        logger.warning("Prediction file empty.")
        return
    
    if "prediction_probability" not in df.columns:
        logger.warning("Prediction probability column missing.")
        return
    
    probs = df["prediction_probability"]
    
    report = {
        "timestamp" : datetime.now(timezone.utc).isoformat(),
        "total_predictions" : int(len(df)),
        "avg_probability" : float(probs.mean()),
        "std_probability" : float(probs.std()),
        "min_probability" : float(probs.min()),
        "max_probability" : float(probs.max()),
        "high_risk_ratio" : float((probs > 0.7).mean()),
        "low_risk_ratio" : float((probs < 0.3).mean()),
    }
    
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info("Prediction monitoring report generated.")
    logger.info(report)
    
    
if __name__ == "__main__":
    generate_prediction_report()
