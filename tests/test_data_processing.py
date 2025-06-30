# tests/test_data_processing.py

import pandas as pd
from src.utils import calculate_rfm  # assuming you moved RFM code to a helper

def test_rfm_columns():
    sample_data = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "TransactionStartTime": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-05"]),
        "TransactionId": [1, 2, 3],
        "Amount": [100, 150, 200]
    })
    snapshot = pd.to_datetime("2024-01-15")
    result = calculate_rfm(sample_data, snapshot)
    assert "Recency" in result.columns
    assert "Frequency" in result.columns
    assert "Monetary" in result.columns
