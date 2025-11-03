import os

class ModelRegistry:
    def __init__(self):
        model_dir = os.getenv("MODEL_DIR", "models/")
        self.models = {
            "random_forest": os.path.join(model_dir, "random_forest"),
            "xgboost": os.path.join(model_dir, "xgboost"),
            "neural_network": os.path.join(model_dir, "neural_network")
        }
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")

    def get_model_path(self, model_type: str):
        """Return the path for the requested model type."""
        path = self.models.get(model_type)
        if path and os.path.exists(path):
            return path
        return None
