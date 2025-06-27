import pytest
import pandas as pd
from src.data_processing import load_and_process_data

def test_load_data_returns_dataframe():
    filepath = "data/raw/data.xlsx"
    df_processed, df_original = load_and_process_data(filepath)
    assert isinstance(df_processed, pd.DataFrame)
    assert "AccountId" in df_processed.columns

def test_processed_shape():
    filepath = "data/raw/data.xlsx"
    df_processed, _ = load_and_process_data(filepath)
    # Example: expect more than 50 columns after processing
    assert df_processed.shape[1] > 50
