# Déploiement sur Azure

Le backend peut être déployé sur **Azure App Service** via :

- Docker image
- GitHub Actions (workflow de déploiement)

## Étapes générales

1. Créer un Azure Container Registry
2. Push des images Docker 
3. Lier le registre à Azure App Service
4. Déployer automatiquement via CI/CD