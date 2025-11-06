import os
import time
import json
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from src.api.utils.cache_utils import load_cache, save_cache

load_dotenv()
router = APIRouter(prefix="/models", tags=["Model Management"])

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "my_model")
CACHE_FILE = os.getenv("MODEL_VERSIONS_CACHE_FILE", "src/api/cache/model_versions.json")
CACHE_TTL = int(os.getenv("CACHE_TTL", 120))
MODEL_DIR = os.getenv("MODEL_DIR", "models/")

# In-memory cache
_cache = {"data": None, "last_update": 0}

# Initializing MLflow client
mlflow_client = None
if MLFLOW_URI:
    try:
        mlflow_client = MlflowClient(tracking_uri=MLFLOW_URI)
    except Exception:
        mlflow_client = None

# Helper functions
def fetch_models_from_mlflow():
    """Fetching model versions from MLflow tracking server."""
    if not mlflow_client:
        raise HTTPException(status_code=500, detail="MLflow tracking URI not configured or unreachable")
    try:
        versions = mlflow_client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        result = []
        for v in versions:
            result.append({
                "model_name": REGISTERED_MODEL_NAME,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "source": v.source,
                "creation_timestamp": v.creation_timestamp,
                "last_updated_timestamp": v.last_updated_timestamp
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models from MLflow: {str(e)}")


def fetch_models_from_local():
    """Fetching models from local model directory."""
    if not os.path.exists(MODEL_DIR):
        return []

    models = []
    for model_name in os.listdir(MODEL_DIR):
        model_path = os.path.join(MODEL_DIR, model_name)
        if os.path.isdir(model_path):
            metadata_file = os.path.join(model_path, "metadata.json")
            info = {"model_name": model_name, "path": model_path, "source": "local"}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        info.update(json.load(f))
                except Exception:
                    pass
            models.append(info)
    return models


def load_cached_models():
    """Loading models from cache if valid, otherwise refresh cache."""
    now = time.time()
    if _cache["data"] and (now - _cache["last_update"]) < CACHE_TTL:
        return _cache["data"]

    if os.path.exists(CACHE_FILE):
        cached = load_cache(CACHE_FILE, default=[])
        _cache["data"] = cached
        _cache["last_update"] = now
        return cached

    return refresh_model_cache()


def refresh_model_cache():
    """Refresh the model cache from MLflow (with local fallback)."""
    try:
        models = fetch_models_from_mlflow()
    except HTTPException:
        # Fallback to local models if MLflow fails
        models = fetch_models_from_local()

    save_cache(CACHE_FILE, models)
    _cache["data"] = models
    _cache["last_update"] = time.time()
    return models

@router.get("/")
def get_all_models():
    """
    Unified endpoint: fetches model information from MLflow or local fallback.
    Returns cached data if recent.
    """
    try:
        data = load_cached_models()
        source = "mlflow" if mlflow_client else "local"
        return {
            "source": source,
            "cached_at": _cache["last_update"],
            "models": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")