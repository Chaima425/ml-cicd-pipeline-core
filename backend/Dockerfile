# Utilise une image officielle Python
FROM python:3.11-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'app
COPY app ./app
COPY ml ./ml
COPY model ./model

# Exposer le port FastAPI
EXPOSE 8000

# Commande par défaut pour lancer l'API
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# FROM python:3.10-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]