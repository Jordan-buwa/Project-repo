from fastapi import HTTPException, APIRouter, Body, status
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.api.utils.customer_data import CustomerData
from src.api.utils.database import get_db_connection
from src.data_pipeline.preprocess import ProductionPreprocessor
from typing import Dict, Any
import os
import joblib
import json
import logging

from src.api.utils.config import APIConfig, get_model_path, get_allowed_model_types
from src.api.utils.response_models import PredictionResponse, FeatureData
from src.api.utils.error_handlers import (
     DataNotFoundError, PreprocessingError,
    handle_model_error, handle_data_error, raise_if_model_not_found
)
from src.api.utils.models_types import ModelType, validate_model_type
from src.api.ml_models import get_model_path, get_model
router = APIRouter(prefix="/predict")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Initialize configuration
config = APIConfig()

def get_latest_model(model_type: str):
    """Get the latest model path using centralized configuration."""
    try:
        if not validate_model_type(model_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type. Must be one of: {ModelType.get_all_types()}"
            )
        
        model_path = get_model_path(model_type)
        raise_if_model_not_found(model_path, model_type)
        return model_path
    except Exception as e:
        handle_model_error(model_type, e)


def load_model(model_path: str, model_type: str):
    normalized_type = model_type.replace("-", "_")
    
    if normalized_type == ModelType.NEURAL_NET:
        import torch
        model = torch.load(model_path)
        model.eval()
        return model
    else:
        return joblib.load(model_path)

def map_dropdowns(payload: dict) -> dict:
    # PRIZM
    prizm = payload.pop("prizm_cluster", None)
    if prizm:
        payload["prizmrur"] = 1 if prizm == "rural" else 0
        payload["prizmub"]  = 1 if prizm == "urban" else 0
        payload["prizmtwn"] = 1 if prizm == "town"  else 0

    # OCCUPATION
    occ = payload.pop("occupation", None)
    occ_map = {
        "professional": "occprof",
        "clerical": "occcler",
        "craft": "occcrft",
        "student": "occstud",
        "homemaker": "occhmkr",
        "retired": "occret",
        "self-employed": "occself"
    }
    for k in occ_map.values():
        payload[k] = 0
    if occ and occ in occ_map:
        payload[occ_map[occ]] = 1

    # MARITAL
    marital = payload.pop("marital_status", "unknown")
    payload["marryyes"] = 1 if marital == "married" else 0
    payload["marryun"] = 1 if marital == "unmarried" else 0

    return payload

@router.post("/{model_type}", response_model=PredictionResponse)
def predict_from_payload(
    model_type: str,
    payload: Dict[str, Any] = Body(..., example={
        "revenue": 45.3, "mou": 120.5, "months": 12, "credita": "A",
    })
):
    """
    Accept raw customer data → predict churn.
    """
    payload = map_dropdowns(payload)
    if model_type not in get_allowed_model_types():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"model_type must be one of: {ModelType.get_all_types()}"
        )

    try:
        raw_data = payload

        # Check if churn value is provided
        provided_churn = raw_data.get('churn')
        customer_id = raw_data.get('customer_id', 'ad-hoc')

        # If churn is provided, return it as the prediction
        if provided_churn is not None:
            churn_value = float(provided_churn)  # Convert to float for consistency
            prediction_data = {
                "model_type": model_type,
                "prediction": churn_value,
                "model_path": "provided_in_input",
                "customer_id": customer_id,
                "preprocessing_applied": False,
                "confidence": 1.0,
                "note": "Using provided churn value from input"
            }

            return PredictionResponse(
                status="success",
                message="Using provided churn value as prediction",
                data=prediction_data
            )    
        df = pd.DataFrame([raw_data])
        
        artifact_path = config.preprocessing_artifacts_path
        if not os.path.exists(artifact_path):
            raise PreprocessingError(f"Preprocessing artifacts not found at {artifact_path}")
        
        processor = ProductionPreprocessor(artifacts_path=artifact_path)
        df_processed = processor.preprocess(df)
        feature_names = processor.get_feature_names()
        features_dict = df_processed[feature_names].iloc[0].to_dict()
        X = [list(features_dict.values())]

        # Load model & predict
        model_path = get_latest_model(model_type)
        model = load_model(model_path, model_type)

        if model_type.replace("-", "_") == ModelType.NEURAL_NET:
            import torch
            tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                pred = model(tensor).cpu().numpy()[0]
        else:
            pred = model.predict(X)[0]

        prediction_data = {
            "model_type": model_type,
            "prediction": float(pred) if isinstance(pred, (int, float, np.generic)) else pred.tolist(),
            "model_path": model_path,
            "customer_id": raw_data.get("customer_id", "ad-hoc"),
            "preprocessing_applied": True
        }

        return PredictionResponse(
            message="Prediction successful",
            data=prediction_data
        )
        
    except Exception as e:
        logger.error(f"Prediction failed for model {model_type}: {str(e)}")
        if "preprocessing" in str(e).lower():
            raise PreprocessingError(f"Failed to preprocess data: {str(e)}")
        else:
            handle_model_error(model_type, e)

@router.get("/{model_type}/customer/{customer_id}", response_model=PredictionResponse)
def predict_from_db_customer(
    model_type: str,
    customer_id: str,
):
    """Predict churn for a specific customer from database."""
    if model_type not in get_allowed_model_types():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"model_type must be one of: {get_allowed_model_types()}"
        )

    try:
        with get_db_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT features FROM customer_data WHERE customer_id = %s",
                (customer_id,)
            )
            row = cur.fetchone()
            if not row:
                raise DataNotFoundError(f"customer_id: {customer_id}")
            
            # Handle both JSON string and already parsed JSON
            features_dict = row["features"]
            if isinstance(features_dict, str):
                try:
                    features_dict = json.loads(features_dict)
                except json.JSONDecodeError:
                    raise PreprocessingError(f"Invalid features format for customer {customer_id}")
            
            # Convert to 2D list for model
            X = [list(features_dict.values())]

            # Load model
            model_path = get_latest_model(model_type)
            model = load_model(model_path, model_type)

            # Predict
            if model_type == "neural-net":
                import torch
                tensor = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    pred = model(tensor).cpu().numpy()[0]
            else:
                pred = model.predict(X)[0]

            prediction_data = {
                "model_type": model_type,
                "prediction": float(pred),
                "model_path": model_path,
                "customer_id": customer_id,
                "feature_count": len(features_dict),
                "preprocessing_applied": False  # Data already preprocessed
            }

            return PredictionResponse(
                message=f"Prediction successful for customer {customer_id}",
                data=prediction_data
            )
            
    except Exception as e:
        logger.error(f"Prediction failed for customer {customer_id}: {str(e)}")
        if "not found" in str(e).lower():
            handle_data_error(f"customer_id: {customer_id}", e)
        else:
            handle_model_error(model_type, e)
@router.get("/{model_type}/batch/{batch_id}", response_model=PredictionResponse)
def predict_from_db_batch(
    model_type: str,
    batch_id: str,
    limit: int = 100,
):
    """Return predictions for the *first N* records of a batch."""
    if model_type not in get_allowed_model_types():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"model_type must be one of: {get_allowed_model_types()}"
        )

    try:
        with get_db_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """SELECT customer_id, features
                   FROM customer_data
                   WHERE batch_id = %s
                   ORDER BY created_at
                   LIMIT %s""",
                (batch_id, limit)
            )
            rows = cur.fetchall()
            if not rows:
                raise DataNotFoundError(f"batch_id: {batch_id}")

        results = []
        model_path = get_latest_model(model_type)
        model = load_model(model_path, model_type)

        for r in rows:
            X = [list(r["features"].values())]
            if model_type == "neural-net":
                import torch
                tensor = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    pred = model(tensor).cpu().numpy()[0]
            else:
                pred = model.predict(X)[0]

            results.append({
                "customer_id": r["customer_id"],
                "prediction": pred.tolist() if isinstance(pred, (list, tuple)) else float(pred),
            })

        prediction_data = {
            "model_type": model_type,
            "batch_id": batch_id,
            "predictions": results,
            "model_path": model_path,
            "prediction_count": len(results),
            "preprocessing_applied": False  # Data already preprocessed
        }

        return PredictionResponse(
            message=f"Batch prediction successful for {len(results)} customers",
            data=prediction_data
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed for batch {batch_id}: {str(e)}")
        if "not found" in str(e).lower():
            handle_data_error(f"batch_id: {batch_id}", e)
        else:
            handle_model_error(model_type, e)

