from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    body = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert "prediction" in response.json()
