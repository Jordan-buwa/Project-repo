import yaml
import json
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
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
# Importing dataframe validation function
from src.data_pipeline.pipeline_data import fetch_preprocessed

# Suppressing unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Loading environment variables
load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

# Loading training config
with open("config/config_train_rf.yaml", "r") as f:
    train_config = yaml.safe_load(f)

# Logging setup - ADJUSTED: Use config values
os.makedirs(os.path.dirname(train_config["logging"]["log_path"]), exist_ok=True)
logging.basicConfig(
    filename=train_config["logging"]["log_path"],
    level=getattr(logging, train_config["logging"]["log_level"]),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_COL = train_config["data"]["target_column"]
MODEL_DIR = train_config.get("output", {}).get("model_dir", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

if os.path.exists("data/processed/processed_data.csv"):
    df = pd.read_csv("data/processed/processed_data.csv")
else: df = fetch_preprocessed()

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
mlflow.set_experiment(train_config["experiment_name"])  # ADJUSTED: Use config value
logger.info(f"MLflow tracking URI: {MLFLOW_URI}")

# Tagging MLflow run with DVC data hash for lineage
dvc_hash = subprocess.getoutput("dvc hash data/processed/preprocessed.csv")

# Defining models
available_models = {
    "random_forest": RandomForestClassifier(random_state=train_config["model"]["random_state"])
}

def save_data_schema(X: pd.DataFrame, y: pd.Series, target_col: str, model_type: str, model_version: str):
    """Save data schema both locally and to MLflow with organized structure"""
    schema = {
        "model_type": model_type,
        "model_version": model_version,
        "required_columns": X.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
        "target_column": target_col,
        "schema_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "feature_count": X.shape[1],
        "sample_count": X.shape[0],
        "class_distribution": y.value_counts().to_dict(),
        "training_data_hash": dvc_hash
    }
    
    # Saving schema locally in organized structure
    schemas_dir = os.path.join(MODEL_DIR, model_type, "schemas")
    os.makedirs(schemas_dir, exist_ok=True)
    local_schema_path = os.path.join(schemas_dir, f"{model_version}_schema.json")
    
    with open(local_schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Schema saved locally at: {local_schema_path}")
    
    # Saving schema to MLflow
    mlflow_schema_path = "schema.json"
    with open(mlflow_schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    mlflow.log_artifact(mlflow_schema_path, "schema")
    logger.info(f"Schema saved to MLflow artifacts")
    
    # Cleaning up temp file
    if os.path.exists(mlflow_schema_path):
        os.remove(mlflow_schema_path)
    
    return schema, local_schema_path

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
            pre_dispatch="2 * n_jobs",
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
            
            # Logging parameters
            mlflow.log_param("num_samples", X.shape[0])
            mlflow.log_param("num_features", X.shape[1])
            mlflow.log_param("target_column", TARGET_COL)

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

            # Storing best models passing thresholds
            #if all(avg_metrics.get(m, 0) >= t for m, t in thresholds.items()):
            best_models[name] = (best_model, avg_metrics)
                #logger.info(f"Model {name} passed thresholds: {avg_metrics}")
            #else:
                #logger.warning(f"Model {name} did not meet thresholds: {avg_metrics}")

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Organizing directory structure
    model_type = "random_forest"
    versions_dir = os.path.join(MODEL_DIR, model_type, "versions")
    os.makedirs(versions_dir, exist_ok=True)
    
    # Creating version name
    model_version_name = f"random_forest_churn_v{timestamp}"
    joblib.dump(best_model, os.path.join(MODEL_DIR, model_version_name))
    local_model_path = os.path.join(versions_dir, f"{model_version_name}.joblib")
    joblib.dump(best_model, local_model_path)
    logger.info(f"Best model saved locally at: {local_model_path}")
    
    # Saving schema to both locations
    schema, local_schema_path = save_data_schema(X, y, TARGET_COL, model_type, model_version_name)
    
    # Logging model to MLflow with model type in name
    mlflow_registry_name = f"{model_type}_churn_model"
    mlflow.sklearn.log_model(
        best_model, 
        artifact_path="model",
        registered_model_name=mlflow_registry_name,
        input_example=X.iloc[:5]
    )
    
    # Logging additional metadata
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("model_version", model_version_name)
    mlflow.log_param("local_schema_path", local_schema_path)
    mlflow.log_param("local_model_path", local_model_path)
    mlflow.log_artifact(local_model_path, "local_model")
    
    logger.info(f"Model registered in MLflow as '{mlflow_registry_name}'")
    logger.info(f"Training completed. Best model: {best_model_name} with metrics: {best_metrics}")
    
    # Printing summary
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Model Type: {model_type}")
    print(f"Model Version: {model_version_name}")
    print(f"Local Model Path: {local_model_path}")
    print(f"Local Schema Path: {local_schema_path}")
    print(f"MLflow Model: {mlflow_registry_name}")
    print(f"Performance Metrics: {best_metrics}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
else:
    logger.error("No model met the performance thresholds")
    print("Training failed: No model met the performance thresholds")