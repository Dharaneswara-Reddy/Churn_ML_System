from tracemalloc import start
from wsgiref import validate
from fastapi import FastAPI, HTTPException
import pickle
import time
import pandas as pd


from churn_system.schema import validate_inference_data
from churn_system.config.config import load_config
from churn_system.logging.logger import get_logger
from churn_system.monitoring.prediction_store import store_prediction
from pathlib import Path

logger = get_logger(__name__)

config = load_config()

app = FastAPI(title="Churn Prediction API")

# Load the model Once at startup

model_path = Path(config["model"]["production_path"])

with open(model_path,"rb") as f:
    model = pickle.load(f)

THRESHOLD = config["inference"]["threshold"]    

@app.get("/")
def health_check():
    return {"status" : "ok", "message" : "Churn model is running"}

@app.post("/predict")
def predict(payload : dict):
    """
    
    Accepts raw feature dictionary and returns churn probability
    
    """
    start_time = time.time()
    logger.info("Received prediction request")
    
    try:
        df = pd.DataFrame([payload])
        df_valid = validate_inference_data(df)
    except Exception as e:
        logger.error(f'Validation failed: {e}')
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        prob = model.predict_proba(df_valid)[:,1][0]
        prediction = int(prob >= THRESHOLD)
        store_prediction(df_valid, prob, prediction)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail = "Prediction failed")
    
    latency = time.time() - start_time
    
    logger.info(
        f"Prediction made | prob = {prob:.4f} | pred = {prediction} | latency = {latency:.4f}s"
    )
    
    
    return {
            "churn_probability" : round(float(prob), 4),
            "prediction" : prediction,
            "threshold" : THRESHOLD,
            "latency_seconds" : round(latency,4)
        }