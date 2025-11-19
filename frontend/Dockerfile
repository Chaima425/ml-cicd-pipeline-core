FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# # Définir la variable d'environnement pour le backend
# ENV BACKEND_URL=http://iris-backend:8000

# Exposer le port Streamlit
EXPOSE 8501

# Commande par défaut pour lancer l'app Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

