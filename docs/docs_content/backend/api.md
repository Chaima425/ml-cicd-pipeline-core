# API Backend

Cette documentation décrit l'API FastAPI pour le projet CI/CD IA.

---

## Base URL

```text
http://localhost:8000
```


## Endpoints

### 1. Vérification de l'API

GET /

Description : Permet de vérifier que l'API est en ligne.

Paramètres : Aucun

Réponse :

{
  "status": "ok"
}

### 2. Prédiction Iris

POST /predict

Description : Prédit la classe de l'Iris à partir de ses caractéristiques.

Paramètres (JSON) :

Champ	Type	Description
sepal_length	float	Longueur du sépale
sepal_width	float	Largeur du sépale
petal_length	float	Longueur du pétale
petal_width	float	Largeur du pétale

### Exemple de requête :

POST /predict
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}


Exemple de réponse :

{
  "prediction": 0
}


### Note : La valeur prediction correspond à l'index de la classe prédite par le modèle.

### Chargement du modèle

Le modèle est chargé automatiquement depuis le fichier model/model.pkl.
Si le fichier n'existe pas, une erreur est levée.

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.pkl"

### Logging

L'API utilise le module logging pour suivre les événements et les erreurs.

### Niveau de log : INFO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
