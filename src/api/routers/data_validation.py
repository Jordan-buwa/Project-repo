import os
import json
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, FastAPI, Query
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from src.api.utils.validation_utils import validate_against_schema, get_data_version
from src.api.utils.cache_utils import load_cache, save_cache
from src.api.routers.model_registry import ModelRegistry

load_dotenv()

router = APIRouter(prefix="/api/data_validation", tags=["Data Validation"])

app = FastAPI()
# Config & cache
CACHE_FILE = os.getenv("DATA_VALIDATION_CACHE_FILE", "src/api/cache/data_validation_cache.json")
_validation_cache = load_cache(CACHE_FILE, default=[])
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

# MLflow client
mlflow_registry = ModelRegistry()
mlflow_client = MlflowClient(tracking_uri=mlflow_registry.mlflow_tracking_uri)

def _log_validation_mlflow(result: dict):
    import mlflow
    mlflow.set_tracking_uri(mlflow_registry.mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_registry.experiment_name)
    with mlflow.start_run(run_name=f"data_validation_{datetime.now().isoformat()}"):
        for key, value in result.items():
            if isinstance(value, (int, float, str)):
                mlflow.log_param(key, value)
        if result.get("issues"):
            mlflow.log_param("issues", json.dumps(result["issues"]))
        mlflow.set_tag("validation_stage", result.get("stage", "unknown"))

@app.post("/validate")
async def validate_dataset(
    file: UploadFile = File(...),
    experiment: str = Query(None),
    model_type: str = Query("random_forest")
):
    model_path = mlflow_registry.get_model_path(model_type)
    if not model_path:
        raise HTTPException(status_code=400, detail=f"Model '{model_type}' not found in MODEL_DIR")

    try:
        df = pd.read_csv(file.file)
        issues = validate_against_schema(df)  # validates schema
        stage = "Validated" if not issues else "Failed"
        result = {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "issues": issues,
            "experiment": experiment or mlflow_registry.experiment_name,
            "model_version": model_type,
            "data_version": get_data_version()
        }
        _validation_cache.append(result)
        save_cache(CACHE_FILE, _validation_cache)
        _log_validation_mlflow(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/history")
def get_validation_history(
    stage: str = Query(None),
    sort_by: str = Query("timestamp"),
    sort_order: str = Query("desc"),
    limit: int = Query(10, ge=1),
    offset: int = Query(0, ge=0)
):
    try:
        results = _validation_cache.copy()
        if stage:
            stage = stage.capitalize()
            if stage not in ["Validated", "Failed"]:
                raise HTTPException(status_code=400, detail="Invalid stage value")
            results = [r for r in results if r["stage"] == stage]
        if sort_by not in ["timestamp", "rows"]:
            raise HTTPException(status_code=400, detail="Invalid sort_by value")
        reverse = sort_order.lower() == "desc"
        results.sort(key=lambda r: r[sort_by], reverse=reverse)
        paginated = results[offset: offset + limit]
        return {"total_results": len(results), "limit": limit, "offset": offset, "results": paginated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving validation history: {str(e)}")
