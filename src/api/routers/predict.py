from fastapi import HTTPException, APIRouter, Body
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.data_pipeline.preprocess import setup_logger
from src.api.utils.customer_data import CustomerData
from src.api.utils.database import get_db_connection
from src.data_pipeline.preprocess import ProductionPreprocessor
from typing import Dict, Any
import os
import joblib
import json

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = setup_logger("predict_router")

def get_latest_model(model_type: str):
    model_ext = ".pth" if model_type == "neural-net" else ".joblib"
    model_files = [
        f for f in os.listdir("models")
        if f.endswith(model_ext) and model_type.replace("-", "") in f.lower()
    ]
    if not model_files:
        raise HTTPException(
            status_code=404, detail=f"No trained {model_type} model found."
        )
    latest_model = max(
        model_files, key=lambda f: os.path.getmtime(os.path.join("models", f))
    )
    return os.path.join("models", latest_model)


def load_model(model_path: str, model_type: str):
    if model_type == "neural-net":
        import torch
        model = torch.load(model_path)
        model.eval()
        return model
    else:
        return joblib.load(model_path)


@router.post("/{model_type}")
def predict_from_payload(
    model_type: str,
    payload: CustomerData = Body(..., example={
        "revenue": 45.3, "mou": 120.5, "months": 12, "credita": "A",
    })
):
    """
    Accept raw customer data → run full preprocessing → predict churn.
    """
    if model_type not in {"xgboost", "random-forest", "neural-net"}:
        raise HTTPException(status_code=400,
                            detail="model_type must be xgboost|random-forest|neural-net")

    raw_data = payload.dict()
    df = pd.DataFrame([raw_data])
    artifact_path = "src/data_pipeline/preprocessing_artifacts.json"
    if not os.path.exists(artifact_path):
        raise HTTPException(status_code=500, detail="Preprocessing artifacts not found")
    

    processor = ProductionPreprocessor(artifacts_path = artifact_path)
    df_processed = processor.preprocess(df)
    feature_names = processor.get_feature_names()
    features_dict = df_processed[feature_names].iloc[0].to_dict()
    X = [list(features_dict.values())]

    # 7. Load model & predict
    model_path = get_latest_model(model_type)
    model = load_model(model_path, model_type)

    if model_type == "neural-net":
        import torch
        tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred = model(tensor).cpu().numpy()[0]
    else:
        pred = model.predict(X)[0]

    return {
        "model_type": model_type,
        "prediction": float(pred) if isinstance(pred, (int, float, np.generic)) else pred.tolist(),
        "model_path": model_path,
        "customer_id": raw_data.get("customer_id", "ad-hoc"),
        "preprocessing_applied": True
    }

@router.get("/{model_type}/customer/{customer_id}")
def predict_from_db_customer(
    model_type: str,
    customer_id: str,
):

    with get_db_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT features FROM customer_data WHERE customer_id = %s",
            (customer_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Handle both JSON string and already parsed JSON
        features_dict = row["features"]
        if isinstance(features_dict, str):
            try:
                features_dict = json.loads(features_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Invalid features format")
        

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

        return {
            "model_type": model_type,
            "prediction": float(pred),
            "model_path": model_path,
            "customer_id": customer_id,
            "feature_count": len(features_dict)
}
@router.get("/{model_type}/batch/{batch_id}")
def predict_from_db_batch(
    model_type: str,
    batch_id: str,
    limit: int = 100,
):
    """Return predictions for the *first N* records of a batch."""
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
            raise HTTPException(status_code=404,
                                detail="Batch empty or not found")

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

    return {
        "model_type": model_type,
        "batch_id": batch_id,
        "predictions": results,
        "model_path": model_path,
    }

