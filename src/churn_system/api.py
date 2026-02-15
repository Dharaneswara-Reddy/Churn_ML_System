from wsgiref import validate
from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd


from churn_system.schema import validate_inference_data
from churn_system.config import load_config
from pathlib import Path


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
    
    try:
        df = pd.DataFrame([payload])
        df_valid = validate_inference_data(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        prob = model.predict_proba(df_valid)[:,1][0]
        prediction = int(prob >= THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail = "Prediction failed")
    
    
    return {
            "churn_probability" : round(float(prob), 4),
            "prediction" : prediction,
            "threshold" : THRESHOLD
        }