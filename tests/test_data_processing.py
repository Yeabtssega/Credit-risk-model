import os
import pytest
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def file_exists_or_skip(filepath):
    if not os.path.exists(filepath):
        pytest.skip(f"Skipping test because {filepath} is missing")

def test_data_shape():
    path = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
    file_exists_or_skip(path)
    df = pd.read_csv(path, header=None)
    assert df.shape[0] > 0, "Processed data CSV is empty."

def test_no_nulls():
    path = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
    file_exists_or_skip(path)
    df = pd.read_csv(path, header=None)
    assert df.isnull().sum().sum() == 0, "Processed data CSV contains null values."

def test_target_shape():
    path = os.path.join(BASE_DIR, "data", "processed", "target_labels.csv")
    file_exists_or_skip(path)
    df = pd.read_csv(path)
    assert df.shape[0] > 0, "Target labels CSV is empty."
