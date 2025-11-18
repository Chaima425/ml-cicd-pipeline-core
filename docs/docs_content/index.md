# Documentation du projet CI/CD IA

Bienvenue dans la documentation du projet **Mise en place d’un pipeline CI/CD**.

Ce projet inclut :
- Entraînement d’un modèle simple (MLflow + scikit-learn)
- API FastAPI (`/` et `/predict`)
- Frontend Streamlit minimal
- Tests unitaires
- Documentation déployée avec MkDocs
- Dockerisation backend & frontend
- Pipeline GitHub Actions pour build et push des images Docker
- Déploiement sur Azure App Service

## Structure du projet
```
ml-cicd-pipeline-core
├──backend
│   ├──app
│   │   ├──api.py
│   │   └──main.py
│   ├──ml
│   │   └──train.py
│   ├──model
│   │   ├──iris_model_ef02ee4c0aa14dadbe3bd7f96cb39016.pkl
│   │   └──model.pkl
│   ├──tests
│   │   └──test_api.py
│   ├──Dockerfile
│   └──requirements.txt
├──docs
│   ├──docs
│   │   └──index.md
│   └──mkdocs.yml
├──frontend
│   ├──app.py
│   ├──Dockerfile
│   └──requirements.txt
├──.github
│   └──workflows
│   │   └──ci.yml
├──pyproject.toml
└──.gitignore
```

## Sections de la documentation

- [Backend](backend/api.md) – API, modèle, tests  
- [Frontend](frontend/app.md) – Application Streamlit  
- [CI/CD](cicd/pipeline.md) – Pipeline GitHub Actions et Docker  
- [MLflow](mlflow/tracking.md) – Suivi des expériences et modèles  
- [Deploy](deploy/azure.md) – Déploiement sur Azure et DockerHub


