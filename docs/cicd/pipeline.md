# Pipeline CI/CD (GitHub Actions)

Le pipeline CI/CD permet :

- installation des dépendances backend & frontend
- exécution des tests
- build Docker
- push sur un registre (Docker Hub ou Azure)
- déploiement

## Objectifs

1. Automatiser la qualité (tests)
2. Automatiser le packaging (Docker)
3. Automatiser le déploiement

Un fichier `ci.yml` sera présent dans :
`.github/workflows/ci.yml`