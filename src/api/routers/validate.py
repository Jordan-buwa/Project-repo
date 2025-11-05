# src/api/routers/validate.py
import os
import json
import mlflow
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, FastAPI, Depends
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional

from src.api.utils.cache_utils import load_cache, save_cache
from src.api.utils.validation_utils import (
    fetch_data, 
    validate_schema, 
    get_allowed_model_types
)

load_dotenv()

router = APIRouter(prefix="/data_validation")

# Configuration - will be reloaded for each request
def get_config():
    """Reload configuration from environment for each request."""
    load_dotenv(override=True)  # This reloads environment variables
    return {
        "CACHE_FILE": os.getenv("DATA_VALIDATION_CACHE_FILE", "src/api/cache/data_validation_cache.json"),
        "VALIDATION_CONFIG": os.getenv("VALIDATION_CONFIG", "config/config_api_data-val.yaml"),
        "MODEL_DIR": os.getenv("MODEL_DIR", "models/")
    }

def get_validation_cache(config: dict = Depends(get_config)):
    """Dependency to get validation cache."""
    cache_file = config["CACHE_FILE"]
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    return load_cache(cache_file, default=[])

@router.post("/validate")
async def validate_dataset(
    db_connection_string: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    csv_path: str = Query("data/production/client.csv"),
    db_delay_seconds: int = Query(600),
    max_rows: int = Query(100),
    model_type: str = Query(..., description="Model type to validate against"),
    model_version: str = Query(..., description="Version of the model for schema matching"),
    config: dict = Depends(get_config),
    validation_cache: list = Depends(get_validation_cache)
):
    """
    Fetch data (DB or CSV) and validate against schema linked to model_type & model_version.
    """
    try:
        # Use config from dependency
        CACHE_FILE = config["CACHE_FILE"]
        VALIDATION_CONFIG = config["VALIDATION_CONFIG"]
        MODEL_DIR = config["MODEL_DIR"]

        print(f"Using MODEL_DIR: {MODEL_DIR}")  # Debug print
        print(f"Using VALIDATION_CONFIG: {VALIDATION_CONFIG}")  # Debug print

        # Checking allowed model types dynamically from config or registered models
        allowed_models = get_allowed_model_types(VALIDATION_CONFIG, model_dir=MODEL_DIR)
        if model_type not in allowed_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model_type. Must be one of {allowed_models}"
            )

        # Fetching data from database or local CSV
        df, source = fetch_data(
            db_connection_string=db_connection_string,
            query=query,
            csv_path=csv_path,
            db_delay_seconds=db_delay_seconds,
            max_rows=max_rows
        )

        # Validating schema using the dynamic MODEL_DIR from config
        issues = validate_schema(
            df, 
            model_type=model_type, 
            model_version=model_version, 
            model_dir=MODEL_DIR,  # Use dynamic MODEL_DIR from config
            config_path=VALIDATION_CONFIG  # Pass config for schema discovery
        )
        stage = "Validated" if not issues else "Failed"

        result = {
            "timestamp": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "issues": issues,
            "stage": stage,
            "source": source,
            "model_type": model_type,
            "model_version": model_version
        }

        # Saving to cache
        validation_cache.append(result)
        save_cache(CACHE_FILE, validation_cache)

        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")