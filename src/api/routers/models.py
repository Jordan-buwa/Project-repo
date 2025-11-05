import os
from fastapi import APIRouter, HTTPException, FastAPI
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

router = APIRouter()

class ModelRegistry:
    def __init__(self):
        # Read from environment variable or fallback to local directory
        self.model_dir = os.getenv("MODEL_DIR", "models/")
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)

        # Initializing MLflow client if URI provided
        self.client = MlflowClient(self.mlflow_tracking_uri) if self.mlflow_tracking_uri else None

        # Defining fallback model locations (in local storage)
        self.local_models = {
            "random_forest": os.path.join(self.model_dir, "random_forest"),
            "xgboost": os.path.join(self.model_dir, "xgboost"),
            "neural_network": os.path.join(self.model_dir, "neural_network"),
        }

    def list_local_models(self):
        """List available local models from models directory."""
        available = []
        for name, path in self.local_models.items():
            if os.path.exists(path):
                available.append({
                    "name": name,
                    "path": path,
                    "source": "local"
                })
        return available

    def list_mlflow_models(self):
        """List registered models from MLflow Tracking server."""
        if not self.client:
            return []

        try:
            models = self.client.list_registered_models()
            model_list = []
            for model in models:
                for version in model.latest_versions:
                    model_list.append({
                        "name": model.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "source": version.source,
                        "run_id": version.run_id,
                        "source_type": "mlflow"
                    })
            return model_list
        except Exception as e:
            # Log the error but don't raise exception - allow fallback to local models
            print(f"Warning: Failed to fetch models from MLflow: {str(e)}")
            return []  # Return empty list instead of raising exception

    def get_all_models(self):
        """Combine both MLflow and local fallback models."""
        all_models = []
        
        # Try to get MLflow models (will return empty list if fails)
        mlflow_models = self.list_mlflow_models()
        all_models.extend(mlflow_models)
        
        # Always include local models as fallback
        local_models = self.list_local_models()
        all_models.extend(local_models)
        
        return all_models


# Initialize registry
model_registry = ModelRegistry()


@router.get("/models")
def list_all_models():
    """
    List all available models from MLflow and fallback local directory.
    If MLflow is not configured, only local models are returned.
    """
    try:
        models = model_registry.get_all_models()
        if not models:
            raise HTTPException(status_code=404, detail="No models found in MLflow or local directory.")
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")