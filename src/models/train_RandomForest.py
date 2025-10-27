import os
import yaml
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
from dotenv import load_dotenv

# Importing dataframe validation function
from src.data_pipeline.pipeline_data import fetch_preprocessed

# Loading environment variables
load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
AZURE_CONN_STR = os.getenv("AZURE_CONN_STR")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Logging Setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train_rf.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Loading Configs
with open("config/config_train_rf.yaml", "r") as f:
    train_config = yaml.safe_load(f)

#with open("config/config_process.yaml", "r") as f:
#    process_config = yaml.safe_load(f)

TARGET_COL = train_config["data"]["target_column"]
MODEL_DIR = train_config["output"]["model_dir"] if "output" in train_config else "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Loading and validating preprocessed data
#processed_path = process_config["output_path"]
#df_raw = pd.read_csv(processed_path)
df = fetch_preprocessed()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
logger.info("Validated preprocessed data loaded successfully. Ready for training.")

# MLflow Setup
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Customer Churn Model Training")
logger.info(f"MLflow tracking URI: {MLFLOW_URI}")

# Defining Models
available_models = {
    "random_forest": RandomForestClassifier(random_state=train_config["random_state"]),
}

# Resampler setup (SMOTETomek)
apply_smotetomek = train_config.get("resampling", {}).get("apply_smotetomek", False)
smote_sampler = SMOTETomek(random_state=train_config["random_state"]) if apply_smotetomek else None

# Model evaluation function
def evaluate_models(X, y, config):
    model_names = config["model_selection"]["model_choice"]
    thresholds = config["model_selection"]["performance_threshold"]
    primary_metric = config["model_selection"]["primary_metric"]

    skf = StratifiedKFold(
        n_splits=config["cv"]["n_splits"], shuffle=True, random_state=config["random_state"]
    )
    best_models = {}

    for name in model_names:
        if name not in available_models:
            logger.warning(f"Model {name} not available, skipping.")
            continue

        model = available_models[name]
        param_grid = config["hyperparameters"].get(name, {})

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=config["cv"]["scoring"],
            cv=skf,
            n_jobs=config["cv"]["n_jobs"],
            verbose=config["cv"]["verbose"]
        )

        with mlflow.start_run(run_name=f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            grid.fit(X, y)
            best_model = grid.best_estimator_
            mlflow.log_params(grid.best_params_)

            fold_metrics = []
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                if smote_sampler:
                    X_train_res, y_train_res = smote_sampler.fit_resample(X_train, y_train)
                else:
                    X_train_res, y_train_res = X_train, y_train

                best_model.fit(X_train_res, y_train_res)

                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_prob)
                }
                fold_metrics.append(metrics)

            avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            mlflow.log_metrics(avg_metrics)
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            if all(avg_metrics[m] >= t for m, t in thresholds.items()):
                best_models[name] = (best_model, avg_metrics)
                local_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
                joblib.dump(best_model, local_path)
                logger.info(f"Model {name} passed thresholds and saved at {local_path}")
            else:
                logger.warning(f"Model {name} did not meet thresholds: {avg_metrics}")

    if not best_models:
        logger.warning("No model met the performance thresholds.")
        return None, None, None

    if len(best_models) == 1:
        name = list(best_models.keys())[0]
        return name, best_models[name][0], best_models[name][1]

    best_model_name = max(best_models, key=lambda n: best_models[n][1][primary_metric])
    return best_model_name, best_models[best_model_name][0], best_models[best_model_name][1]

# Running Training
best_model_name, best_model, best_metrics = evaluate_models(X, y, train_config)

# Saving Best Model Locally
if best_model_name:
    local_model_path = os.path.join(MODEL_DIR, f"{best_model_name}.joblib")
    joblib.dump(best_model, local_model_path)
    logger.info(f"Best model saved locally at: {local_model_path}")

# Azure best model Uploading
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
        except Exception:
            logger.warning(f"Azure container '{AZURE_CONTAINER_NAME}' does not exist. Skipping upload.")
    except Exception as e:
        logger.error(f"Azure upload failed for {best_model_name}: {e}")
