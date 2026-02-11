from wsgiref import validate
from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd


from churn_system.schema import validate_inference_data

app = FastAPI(title="Churn Prediction API")

# Load the model Once at startup

with open("models/experiments/churn_model_v1/model.pkl","rb") as f:
    model = pickle.load(f)
    
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
    except Exception as e:
        raise HTTPException(status_code=500, detail = "Prediction failed")
    
    
    return{
        "churn_probability" : round(float(prob), 4)
    }