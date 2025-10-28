import os
import yaml
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
import warnings
import subprocess

# Importing dataframe validation function
from src.data_pipeline.pipeline_data import fetch_preprocessed

# Suppressing unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Loading environment variables
load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
AZURE_CONN_STR = os.getenv("AZURE_CONN_STR")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train_rf.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Loading training config
with open("config/config_train_rf.yaml", "r") as f:
    train_config = yaml.safe_load(f)

TARGET_COL = train_config["data"]["target_column"]
MODEL_DIR = train_config.get("output", {}).get("model_dir", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Loading and validating preprocessed data
df = fetch_preprocessed()
target_matches = df.columns[df.columns.str.strip().str.lower() == TARGET_COL.lower()]
if len(target_matches) == 0:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Available columns: {list(df.columns)}")
TARGET_COL = target_matches[0]

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
logger.info("Validated preprocessed data loaded successfully. Ready for training.")

# Applying SMOTETomek once before CV
apply_smotetomek = train_config.get("resampling", {}).get("apply_smotetomek", False)
smote_sampler = SMOTETomek(random_state=train_config["model"]["random_state"]) if apply_smotetomek else None
if smote_sampler:
    X, y = smote_sampler.fit_resample(X, y)
    logger.info(f"SMOTETomek applied: dataset size after resampling = {X.shape[0]} samples")

# MLflow setup
if not MLFLOW_URI:
    raise ValueError("MLFLOW_TRACKING_URI not found in .env")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Customer Churn Model Training")
logger.info(f"MLflow tracking URI: {MLFLOW_URI}")

# Tagging MLflow run with DVC data hash for lineage
dvc_hash = subprocess.getoutput("dvc hash data/processed/preprocessed.csv")

# Defining models
available_models = {
    "random_forest": RandomForestClassifier(random_state=train_config["model"]["random_state"])
}

# Function to evaluate models
def evaluate_models(X, y, train_config):
    model_names = train_config["model_selection"]["model_choice"]
    thresholds = train_config["model_selection"]["performance_threshold"]
    primary_metric = train_config["model_selection"]["primary_metric"]

    skf = StratifiedKFold(
        n_splits=train_config["cv"]["n_splits"],
        shuffle=True,
        random_state=train_config["model"]["random_state"]
    )

    best_models = {}
    run_id = None

    for name in model_names:
        if name not in available_models:
            logger.warning(f"Model {name} not available, skipping.")
            continue

        model = available_models[name]
        param_grid = train_config["hyperparameters"].get(name, {})

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=train_config["cv"]["scoring"],
            cv=skf,
            n_jobs=train_config["cv"]["n_jobs"],
            verbose=train_config["cv"]["verbose"],
            pre_dispatch=2 * train_config["cv"]["n_jobs"],
            error_score='raise'
        )

        with mlflow.start_run(run_name=f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            # Setting MLflow tags
            script_name = os.path.basename(__file__) if "__file__" in globals() else "notebook"
            mlflow.set_tag("dvc_data_hash", dvc_hash)
            mlflow.set_tag("script_version", script_name)
            mlflow.set_tag("smotetomek_applied", str(apply_smotetomek))
            mlflow.set_tag("random_state", str(train_config["model"]["random_state"]))
            mlflow.set_tag("preprocess_columns", ",".join(train_config.get("preprocessing", {}).get("drop_columns", [])))
            mlflow.log_param("num_samples", X.shape[0])
            mlflow.log_param("num_features", X.shape[1])

            grid.fit(X, y)
            best_model = grid.best_estimator_
            mlflow.log_params(grid.best_params_)

            # CV metrics
            fold_metrics = []
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # No resampling inside folds
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
                }
                fold_metrics.append(metrics)

            avg_metrics = {k: np.nanmean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            mlflow.log_metrics(avg_metrics)

            # Store best models passing thresholds
            if all(avg_metrics.get(m, 0) >= t for m, t in thresholds.items()):
                best_models[name] = (best_model, avg_metrics)
                logger.info(f"Model {name} passed thresholds: {avg_metrics}")
            else:
                logger.warning(f"Model {name} did not meet thresholds: {avg_metrics}")

            run_id = mlflow.active_run().info.run_id

    if best_models:
        best_model_name = max(
            best_models,
            key=lambda n: best_models[n][1][primary_metric]
        )
        return best_model_name, best_models[best_model_name][0], best_models[best_model_name][1], run_id

    return None, None, None, run_id

# Running evaluation
best_model_name, best_model, best_metrics, run_id = evaluate_models(X, y, train_config)

# Saving & logging best model
if best_model_name:
    local_model_path = os.path.join(MODEL_DIR, f"{best_model_name}.joblib")
    joblib.dump(best_model, local_model_path)
    logger.info(f"Best model saved locally at: {local_model_path}")
    mlflow.sklearn.log_model(best_model, name="model", input_example=X.iloc[:5])

    # Registering in MLflow Model Registry
    try:
        mlflow.register_model(f"runs:/{run_id}/model", "customer_churn_model")
        logger.info(f"Model registered in MLflow Registry as 'customer_churn_model'")
    except Exception as e:
        logger.warning(f"Failed to register model: {e}")

# Azure upload
if best_model_name and AZURE_CONN_STR and AZURE_CONTAINER_NAME:
    try:
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        try:
            container_client.get_container_properties()
            blob_name = f"{best_model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(local_model_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            logger.info(f"Uploaded {best_model_name} to Azure Blob Storage as '{blob_name}'")
        except Exception as e:
            logger.warning(f"Azure container '{AZURE_CONTAINER_NAME}' not accessible: {e}")
    except Exception as e:
        logger.exception(f"Azure upload failed for {best_model_name}: {e}")
else:
    logger.info(f"Azure upload skipped: container or connection string missing")
