from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_home()
    response = client.get()
    assert response.status_code == 200
    assert response.json() == {message Credit Risk API is running}

def test_prediction()
    payload = {features [0.5, 1.2, 3.4, 0.9, 2.1]}  # Example input
    response = client.post(predict, json=payload)
    assert response.status_code == 200
    assert risk_probability in response.json()
