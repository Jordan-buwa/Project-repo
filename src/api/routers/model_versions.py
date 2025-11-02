# src/api/routers/model_versions.py
import os
import json
import time
from fastapi import APIRouter, HTTPException
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from src.models.utils.cache_utils import load_cache, save_cache

load_dotenv()

router = APIRouter(prefix="/api/model_versions", tags=["Model Versions"])

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "my_model")
CACHE_FILE = os.getenv("MODEL_VERSIONS_CACHE_FILE", "src/api/cache/model_versions.json")
CACHE_TTL = int(os.getenv("CACHE_TTL", 60))  # seconds

# In-memory cache
_cache = {"data": None, "last_update": 0}

# MLflow client
mlflow_client = MlflowClient(tracking_uri=MLFLOW_URI)


def fetch_model_versions():
    """Fetch model versions from MLflow with stage info."""
    try:
        versions = mlflow_client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        result = []
        for m in versions:
            result.append({
                "version": m.version,
                "stage": m.current_stage,
                "run_id": m.run_id,
                "source": m.source,
                "creation_timestamp": m.creation_timestamp,
                "last_updated_timestamp": m.last_updated_timestamp
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model versions: {str(e)}")


def load_cached_versions():
    """Load model versions from memory or cache if TTL not expired."""
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
    """Fetch latest model versions and save to cache."""
    versions = fetch_model_versions()
    save_cache(CACHE_FILE, versions)
    _cache["data"] = versions
    _cache["last_update"] = time.time()
    return versions


@router.get("/")
def list_model_versions():
    """List all registered model versions with stages."""
    try:
        versions = load_cached_versions()
        return {"models": versions, "cached_at": _cache["last_update"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stages")
def list_model_stages(stage: str = None):
    """
    List models filtered by stage (Production, Staging, Canary).
    """
    try:
        versions = load_cached_versions()
        if stage:
            stage = stage.capitalize()
            versions = [v for v in versions if v["stage"] == stage]
        return {"models": versions, "cached_at": _cache["last_update"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
def refresh_model_versions():
    """Force refresh model versions from MLflow."""
    try:
        updated = update_model_versions_cache()
        return {"message": "Model versions refreshed successfully", "total": len(updated)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
