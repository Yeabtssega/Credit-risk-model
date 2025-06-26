import os
import pandas as pd

# Get absolute path to project root (one level up from tests folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def test_data_shape():
    path = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
    df = pd.read_csv(path, header=None)
    assert df.shape[0] > 0, "Processed data CSV is empty."

def test_no_nulls():
    path = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
    df = pd.read_csv(path, header=None)
    assert df.isnull().sum().sum() == 0, "Processed data CSV contains null values."

def test_target_shape():
    path = os.path.join(BASE_DIR, "data", "processed", "target_labels.csv")
    df = pd.read_csv(path)
    assert df.shape[0] > 0, "Target labels CSV is empty."
