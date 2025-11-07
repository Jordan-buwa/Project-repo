from fastapi import FastAPI, HTTPException, APIRouter, status
import os
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.api.utils.config import APIConfig, get_allowed_model_types, get_model_path
from src.api.utils.response_models import MetricsResponse, TestDataset
from src.api.utils.error_handlers import (
    ModelNotFoundError, DataNotFoundError,
    handle_model_error, handle_data_error, raise_if_model_not_found
)


router = APIRouter(prefix="/metrics")

# Initialize configuration
config = APIConfig()


def get_latest_model(model_type: str):
    """Get the latest model path using centralized configuration."""
    try:
        model_path = get_model_path(model_type)
        raise_if_model_not_found(model_path, model_type)
        return model_path
    except Exception as e:
        handle_model_error(model_type, e)


def load_model(model_path: str, model_type: str):
    if model_type == "neural-net":
        import torch
        model = torch.load(model_path)
        model.eval()
        return model
    else:
        return joblib.load(model_path)


@router.post("/{model_type}", response_model=MetricsResponse)
def get_metrics(model_type: str):
    """Calculate model performance metrics using standardized test data."""
    if model_type not in get_allowed_model_types():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"model_type must be one of: {get_allowed_model_types()}"
        )

    try:
        # Load test data using standardized format
        test_path = config.test_data_path
        if not os.path.exists(test_path):
            raise DataNotFoundError(f"test_input.json at {test_path}")

        test_dataset = TestDataset.from_json_file(test_path)

        X_test = [list(sample.features.values())
                  for sample in test_dataset.samples]
        y_true = [sample.target for sample in test_dataset.samples]

        # Load model
        model_path = get_latest_model(model_type)
        model = load_model(model_path, model_type)

        # Predict
        if model_type == "neural-net":
            import torch
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            with torch.no_grad():
                y_pred = model(X_tensor).numpy()
                # Convert probabilities to binary
                y_pred = [int(p > 0.5) for p in y_pred]
        else:
            y_pred = model.predict(X_test)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }

        # Include ROC AUC only if binary classification
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
        except:
            metrics["roc_auc"] = None

        metrics_data = {
            "model_type": model_type,
            "model_path": model_path,
            "metrics": metrics,
            "test_samples": len(test_dataset.samples)
        }

        return MetricsResponse(
            message=f"Metrics calculated successfully for {model_type}",
            data=metrics_data
        )

    except Exception as e:
        if "test_input.json" in str(e) or "not found" in str(e).lower():
            handle_data_error("test_input.json", e)
        else:
            handle_model_error(model_type, e)
