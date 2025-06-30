from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import mlflow.pyfunc

# Define the input data schema (adjust features according to your model input)
class CustomerData(BaseModel):
    features: list[float]  # or use a more specific schema if you want

# Define the output schema
class RiskPrediction(BaseModel):
    risk_probability: float

app = FastAPI()

# Build absolute model path relative to this file location
project_root = Path(__file__).parent.parent.parent  # go up 3 levels from src/api/main.py
model_dir = project_root / "models" / "mlflow_model"
model_uri = f"file:///{model_dir.as_posix()}"

# Load the MLflow model
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}

@app.post("/predict", response_model=RiskPrediction)
def predict(data: CustomerData):
    # Predict expects 2D array-like input; data.features is a list of floats for one sample
    prediction = model.predict([data.features])
    return {"risk_probability": float(prediction[0])}
