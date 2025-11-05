import os
import json
import time
from fastapi import APIRouter, HTTPException
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from src.api.utils.cache_utils import load_cache, save_cache

load_dotenv()
router = APIRouter(prefix="/model_versions", tags=["Model Versions"])

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
CACHE_FILE = os.getenv("MODEL_VERSIONS_CACHE_FILE", "src/api/cache/model_versions.json")
CACHE_TTL = int(os.getenv("CACHE_TTL", 60))

_cache = {"data": None, "last_update": 0}
mlflow_client = MlflowClient(tracking_uri=MLFLOW_URI)

def fetch_model_versions():
    """Fetch model versions from MLflow or fallback to local MODEL_DIR."""
    try:
        registered_models = os.getenv("REGISTERED_MODELS", "")
        if registered_models:
            models = registered_models.split(",")
        else:
            models = ["customer_churn_rf_model", "customer_churn_xgb_model", "customer_churn_nn_model"]

        result = []
        for model_name in models:
            try:
                versions = mlflow_client.search_model_versions(f"name='{model_name}'")
                for v in versions:
                    result.append({
                        "model_name": model_name,
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "source": v.source,
                        "creation_timestamp": v.creation_timestamp,
                        "last_updated_timestamp": v.last_updated_timestamp
                    })
            except Exception:
                # fallback to local model path
                result.append({
                    "model_name": model_name,
                    "version": "local",
                    "stage": "local",
                    "run_id": None,
                    "source": os.path.join(os.getenv("MODEL_DIR", "models"), model_name),
                    "creation_timestamp": None,
                    "last_updated_timestamp": None
                })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model versions: {str(e)}")

def load_cached_versions():
    now = time.time()
    if _cache["data"] and (now - _cache["last_update"]) < CACHE_TTL:
        return _cache["data"]

    if os.path.exists(CACHE_FILE):
        versions = load_cache(CACHE_FILE, default=[])
        _cache["data"] = versions
        _cache["last_update"] = now
        return versions

    return update_model_versions_cache()

def update_model_versions_cache():
    versions = fetch_model_versions()
    save_cache(CACHE_FILE, versions)
   
