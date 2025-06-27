from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

app = FastAPI()

# Load model from MLflow registry
model = mlflow.pyfunc.load_model("models:/best_model/Production")

# Define the input schema using valid Python identifiers
class PredictionInput(BaseModel):
    Amount: float
    Value: float
    PricingStrategy: float
    FraudResult: float
    TransactionYear: int
    TransactionMonth: int
    TransactionDay: int
    TransactionWeekday: int
    CurrencyCode_UGX: int
    CountryCode_256_0: int
    CountryCode_nan: int
    ProviderId_ProviderId_1: int
    ProviderId_ProviderId_2: int
    ProviderId_ProviderId_3: int
    ProviderId_ProviderId_4: int
    ProviderId_ProviderId_5: int
    ProviderId_ProviderId_6: int
    ProviderId_nan: int
    ProductId_ProductId_1: int
    ProductId_ProductId_10: int
    ProductId_ProductId_11: int
    ProductId_ProductId_12: int
    ProductId_ProductId_13: int
    ProductId_ProductId_14: int
    ProductId_ProductId_15: int
    ProductId_ProductId_16: int
    ProductId_ProductId_19: int
    ProductId_ProductId_2: int
    ProductId_ProductId_20: int
    ProductId_ProductId_21: int
    ProductId_ProductId_22: int
    ProductId_ProductId_23: int
    ProductId_ProductId_24: int
    ProductId_ProductId_27: int
    ProductId_ProductId_3: int
    ProductId_ProductId_4: int
    ProductId_ProductId_5: int
    ProductId_ProductId_6: int
    ProductId_ProductId_7: int
    ProductId_ProductId_8: int
    ProductId_ProductId_9: int
    ProductCategory_airtime: int
    ProductCategory_data_bundles: int
    ProductCategory_financial_services: int
    ProductCategory_movies: int
    ProductCategory_other: int
    ProductCategory_ticket: int
    ProductCategory_transport: int
    ProductCategory_tv: int
    ProductCategory_utility_bill: int
    ChannelId_ChannelId_1: int
    ChannelId_ChannelId_2: int
    ChannelId_ChannelId_3: int
    ChannelId_ChannelId_5: int

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert input to DataFrame
    data = pd.DataFrame([input_data.dict()])

    # Rename back to original model training names
    data.rename(columns={"CountryCode_256_0": "CountryCode_256.0"}, inplace=True)

    # Predict
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
