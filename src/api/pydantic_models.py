from pydantic import BaseModel, conlist


class CustomerData(BaseModel):
    # List of 6 numeric features (adjust length if needed)
    features: conlist(float, min_items=6, max_items=6)


class PredictionResponse(BaseModel):
    risk_probability: float
