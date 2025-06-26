from fastapi import FastAPI, HTTPException
from pydantic_models import CustomerData, PredictionResponse
import mlflow.pyfunc
import numpy as np

app = FastAPI()

# Load best model from MLflow registry or local path
model = mlflow.pyfunc.load_model("models:/CreditRiskBestModel/Production")

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try:
        features = np.array(data.features).reshape(1, -1)
        pred_prob = model.predict(features)
        risk_prob = float(pred_prob[0])
        return PredictionResponse(risk_probability=risk_prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
