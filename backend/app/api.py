# backend/app/api.py
import logging
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.pkl"

def load_model():
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        raise FileNotFoundError("Model file is missing")
    logger.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/")
async def root():
    return {"status": "ok"}

# Définition du schéma Pydantic pour la requête
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
async def predict(features: IrisFeatures):
    logger.info(f"Requête reçue pour prédiction : {features}")

    # Convertir en liste pour le modèle
    input_list = [
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]

    prediction = model.predict([input_list])[0]

    logger.info(f"Prédiction retournée : {prediction}")

    return {"prediction": int(prediction)}