from pydantic import BaseModel

class CustomerData(BaseModel):
    # Define features your model expects
    feature1: float
    feature2: int
    # Add all required fields here

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([self.dict()])

class PredictionResponse(BaseModel):
    risk_probability: float
