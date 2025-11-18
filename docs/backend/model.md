# Modèle Machine Learning

Le modèle est entraîné dans le fichier `train.py` et sauvegardé dans :

- `backend/model/model.joblib` : le modèle ML entraîné
- `mlruns/` : le répertoire de tracking MLflow

## Étapes effectuées

- Chargement des données Iris
- Séparation train/test
- Entraînement d’un `RandomForestClassifier`
- Enregistrement du modèle en .pkl
- Tracking des expériences avec MLflow

Ce modèle est ensuite utilisé par l’API FastAPI.
