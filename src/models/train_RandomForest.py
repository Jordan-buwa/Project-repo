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
MODEL_DIR = train_config["output"]["model_dir"] if "output" in train_config else "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Loading and validating preprocessed data
df = fetch_preprocessed()
if TARGET_COL.lower() not in df.columns.str.strip().str.lower():
    raise ValueError(f"Target column '{TARGET_COL}' not found. Available columns: {list(df.columns)}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
logger.info("Validated preprocessed data loaded successfully. Ready for training.")

# MLflow setup
if not MLFLOW_URI:
    raise ValueError("MLFLOW_TRACKING_URI not found in .env")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Customer Churn Model Training")
logger.info(f"MLflow tracking URI: {MLFLOW_URI}")

# Defining models
available_models = {
    "random_forest": RandomForestClassifier(random_state=train_config["model"]["random_state"])
}

# Resampler setup
apply_smotetomek = train_config.get("resampling", {}).get("apply_smotetomek", False)
smote_sampler = SMOTETomek(random_state=train_config["model"]["random_state"]) if apply_smotetomek else None

# Model evaluation function
def evaluate_models(X, y, train_config):
    model_names = train_config["model_selection"]["model_choice"]
    thresholds = train_config["model_selection"]["performance_threshold"]
    primary_metric = train_config["model_selection"]["primary_metric"]

    skf = StratifiedKFold(
        n_splits=train_config["cv"]["n_splits"], shuffle=True, random_state=train_config["model"]["random_state"]
    )

    best_models = {}

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
            pre_dispatch="2*n_jobs",
            error_score='raise'
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

            # Store only best models passing thresholds
            if all(avg_metrics[m] >= t for m, t in thresholds.items()):
                best_models[name] = (best_model, avg_metrics)
                logger.info(f"Model {name} passed thresholds: {avg_metrics}")
            else:
                logger.warning(f"Model {name} did not meet thresholds: {avg_metrics}")

    if best_models:
        best_model_name = max(
            best_models,
            key=lambda n: best_models[n][1][primary_metric]
        )
        return best_model_name, best_models[best_model_name][0], best_models[best_model_name][1]

    return None, None, None

# Run evaluation
best_model_name, best_model, best_metrics = evaluate_models(X, y, train_config)

# Save & log best model
if best_model_name:
    local_model_path = os.path.join(MODEL_DIR, f"{best_model_name}.joblib")
    joblib.dump(best_model, local_model_path)
    logger.info(f"Best model saved locally at: {local_model_path}")
    mlflow.sklearn.log_model(best_model, name="model", input_example=X.iloc[:5])

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
        except Exception:
            logger.warning(f"Azure container '{AZURE_CONTAINER_NAME}' does not exist. Skipping upload.")
    except Exception as e:
        logger.exception(f"Azure upload failed for {best_model_name}: {e}")
