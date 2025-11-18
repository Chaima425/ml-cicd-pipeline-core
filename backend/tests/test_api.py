# backend/tests/test_api.py
import sys
from pathlib import Path

# Ajouter le dossier backend au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.api import app
from fastapi.testclient import TestClient
import pytest
from pathlib import Path

client = TestClient(app)

# ---------------------------
# Test route GET /
# ---------------------------
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# ---------------------------
# Test route POST /predict
# ---------------------------
def test_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    # Vérifier que le modèle existe
    model_path = Path(__file__).resolve().parents[1] / "model" / "model.pkl"
    assert model_path.exists(), f"Model not found at {model_path}"

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)

# ---------------------------
# Test payload invalide
# ---------------------------
def test_predict_invalid_payload():
    payload = {"wrong_field": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422