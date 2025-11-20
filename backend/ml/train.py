# # backend/ml/train.py
# import argparse  # pour fichier exécutable en ligne de commande (clI)
# import logging
# import os
# from pathlib import Path

# import joblib
# #from azureml.core import Workspace
# import mlflow
# import mlflow.sklearn  # pour autolog & log_model
# from sklearn import datasets
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split

# # ---------------------------
# # Configuration logging
# # ---------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )
# logger = logging.getLogger(__name__)

# # ---------------------------
# # Helpers for paths & MLflow
# # ---------------------------
# def get_model_dir(env_var_name: str = "MODEL_DIR") -> Path:
#     """
#     Retourne le dossier où sauvegarder le modèle.
#     Priorité : variable d'environnement MODEL_DIR -> default relatif backend/model
#     """
#     env_path = os.getenv(env_var_name)
#     if env_path:
#         model_dir = Path(env_path)
#     else:
#         # Ce fichier est backend/ml/train.py -> parents[1] == backend
#         model_dir = Path(__file__).resolve().parents[1] / "model"
#         logger.info(f"Aucune variable d'env MODEL_DIR. Utilisation du chemin par défaut : {model_dir}")
#     model_dir.mkdir(parents=True, exist_ok=True)
#     return model_dir

# def configure_mlflow():#tracking_url: c'est ici qu'on configure mlflow: local ou distant 
#     """
#     Configure MLflow tracking URI à partir de la variable d'environnement MLFLOW_TRACKING_URI,
#     sinon fallback vers un dossier local ./mlruns dans le répertoire courant.
#     """
#     mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
#     if mlflow_uri:
#         logger.info(f"MLflow distant utilisé : {mlflow_uri}")
#         mlflow.set_tracking_uri(mlflow_uri)
#     else:
#         local_store = Path.cwd() / "mlruns"
#         local_store.mkdir(exist_ok=True)
#         # file:// URI pour stockage local
#         mlflow.set_tracking_uri(f"file://{local_store}")
#         logger.info(f"MLflow local utilisé → stockage dans : {local_store}")

#     # Nom d'expérience configurable via env EXPERIMENT_NAME ou défaut
#     exper_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "MLflow Quickstart")
#     mlflow.set_experiment(exper_name)
#     logger.info(f"Experiment MLflow configurée: '{exper_name}' (tracking uri: {mlflow.get_tracking_uri()})")

# # ---------------------------
# # Training pipeline
# # ---------------------------
# def train_and_log(args):
#     # Configure mlflow & paths
#     configure_mlflow()
#     model_dir = get_model_dir()

#     # 1) Préparer les données (Iris)
#     logger.info("Chargement du dataset Iris")
#     X, y = datasets.load_iris(return_X_y=True)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
#     )
#     logger.info(f"Split done: train={len(X_train)} / test={len(X_test)}")

#     # 2) Autologging scikit-learn (facilite étapes 1-3 du quickstart)
#     mlflow.sklearn.autolog()
#     params = {
#         "solver": args.solver,
#         "max_iter": args.max_iter,
#         "multi_class": "auto",
#         "random_state": args.random_state,
#     }

#     # 3) Entraînement et logging manuel (on entoure d'un run pour extra infos/tags)
#     with mlflow.start_run() as run:
#         run_id = run.info.run_id
#         logger.info(f"Démarrage MLflow run id={run_id}")

#         # Log des hyperparamètres manuellement (utile même si autolog est activé)
#         mlflow.log_params(params)

#         # Entraînement
#         logger.info("Entraînement du modèle (LogisticRegression)")
#         model = LogisticRegression(**params)
#         model.fit(X_train, y_train)

#         # Évaluation
#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)
#         prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
#         rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
#         f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

#         logger.info(f"Metrics test -> accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

#         # Log metrics
#         mlflow.log_metric("accuracy", float(acc))
#         mlflow.log_metric("precision_macro", float(prec))
#         mlflow.log_metric("recall_macro", float(rec))
#         mlflow.log_metric("f1_macro", float(f1))

#         # Tag utile (ex.) : quel dataset on a utilisé
#         mlflow.set_tag("dataset", "iris")
#         mlflow.set_tag("script", "backend/ml/train.py")

#         # 4) Sauvegarde du modèle en .pkl dans model_dir (pour usage pipeline & API)
#         pkl_name = f"iris_model_{run_id}.pkl"
#         pkl_path = model_dir / "model.pkl"  # nom constant pour le déploiement (back/front)
#         # On sauvegarde aussi avec run_id pour traçabilité (debug/tracking)
#         pkl_with_run = model_dir / pkl_name

#         joblib.dump(model, pkl_path)
#         joblib.dump(model, pkl_with_run)
#         logger.info(f"Modèle sauvegardé localement: {pkl_path} (et {pkl_with_run})")

#         # Enregistrer l'artéfact .pkl dans MLflow (artifacts)
#         mlflow.log_artifact(str(pkl_path), artifact_path="model_files")
#         logger.info("Artifact .pkl loggé dans MLflow")

#         # Optionnel : log du modèle via flavor sklearn (pour pouvoir le recharger via mlflow.sklearn.load_model)
#         mlflow.sklearn.log_model(sk_model=model, artifact_path="sklearn_model", registered_model_name=None)
#         logger.info("Modèle loggé via mlflow.sklearn.log_model")

#         logger.info(f"Run terminé: {run_id}. Voir MLflow UI pour détails.")

#     # résumé
#     return {
#         "run_id": run_id,
#         "model_path": str(pkl_path),
#         "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
#     }

# # ---------------------------
# # CLI : pour exécuter ce script directement
# # ---------------------------
# def parse_args(): 
#     parser = argparse.ArgumentParser(description="Train Iris model with MLflow tracking and save .pkl")
#     parser.add_argument("--test-size", type=float, default=0.2, help="Taille du test set")
#     parser.add_argument("--random-state", type=int, default=8888, help="Random seed")
#     parser.add_argument("--solver", type=str, default="lbfgs", help="Solver pour LogisticRegression")
#     parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for solver")
#     return parser.parse_args()

# def main(): 
#     args = parse_args()
#     logger.info("Lancement du training script")
#     result = train_and_log(args)
#     logger.info(f"Résultat: run_id={result['run_id']}, model_path={result['model_path']}")
#     logger.info("Terminé.")

# if __name__ == "__main__":
#     main()




# pour azure mlflow: 

from azureml.core import Workspace
import mlflow
import mlflow.sklearn
import os
from pathlib import Path
import logging
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

logger = logging.getLogger(__name__)

# ---------------------------
# Azure ML + MLflow config
# ---------------------------
def configure_mlflow():
    """
    Configure MLflow pour utiliser :
    1) Azure ML tracking server si config.json existe
    2) Sinon fallback vers un MLflow local
    """

    config_path = Path("config.json")

    if config_path.exists():
        logger.info("→ Connexion au Workspace Azure ML via config.json")

        # 1️ Connexion au workspace Azure
        ws = Workspace.from_config()

        # 2️ Récupération du tracking server Azure MLflow
        tracking_uri = ws.get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)

        logger.info(f"MLflow distant (Azure ML) → {tracking_uri}")

    else:
        logger.warning(" Aucun config.json : MLflow local activé")

        local_dir = Path.cwd() / "mlruns"
        local_dir.mkdir(exist_ok=True)

        mlflow.set_tracking_uri(f"file://{local_dir}")

    # définir l’expérience
    exper_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Iris-AzureML")
    mlflow.set_experiment(exper_name)
    logger.info(f"Expérience MLflow = {exper_name}")


# ---------------------------
# Dossier modèles
# ---------------------------
def get_model_dir():
    model_dir = Path(__file__).resolve().parents[1] / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


# ---------------------------
# Training
# ---------------------------
def train_and_log(args):

    configure_mlflow()
    model_dir = get_model_dir()

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    params = {
        "solver": args.solver,
        "max_iter": args.max_iter,
        "multi_class": "auto",
        "random_state": args.random_state,
    }

    #  ATTENTION : pas d'autolog dans Azure → cause des erreurs
    # mlflow.sklearn.autolog()  ← supprimé

    with mlflow.start_run() as run:

        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # → Logs manuels (obligatoire pour Azure)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        mlflow.log_metric("f1_score", float(f1))

        # Sauvegarde du modèle en local
        pkl_path = model_dir / "model.pkl"
        joblib.dump(model, pkl_path)

        # Upload vers Azure MLflow
        mlflow.log_artifact(str(pkl_path), artifact_path="model")

        logger.info(f"Run terminé → ID: {run.info.run_id}")

    return {
        "run_id": run.info.run_id,
        "model_path": str(pkl_path),
        "metrics": {
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1
        }
    }
