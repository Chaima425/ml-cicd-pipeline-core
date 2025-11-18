# **5. frontend/app.md**

```markdown
# Frontend (Streamlit)

Le frontend est une interface utilisateur simple permettant :

- de saisir les données d'entrée,
- d’appeler l’API FastAPI,
- d’afficher la prédiction.

## Structure

frontend/
└── app.py

## Fonctionnement

- Streamlit affiche des champs numériques (caractéristiques Iris)
- La requête POST est envoyée à l’API
- La prédiction est affichée en temps réel