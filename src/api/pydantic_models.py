from pydantic import BaseModel

class CustomerData(BaseModel):
    Amount: float
    Value: float
    PricingStrategy: float
    FraudResult: int

class RiskPrediction(BaseModel):
    risk_probability: float
    is_high_risk: int
