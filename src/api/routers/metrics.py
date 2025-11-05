from fastapi import FastAPI, HTTPException, APIRouter
import os
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

router = APIRouter(prefix = "/metrics")

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
def get_metrics(model_type: str):
    # Load test data from JSON
    test_path = "test_input.json"
    if not os.path.exists(test_path):
        raise HTTPException(
            status_code=404, detail="test_input.json not found"
        )

    with open(test_path, "r") as f:
        test_data = json.load(f)

    X_test = [list(sample["features"].values()) for sample in test_data]
    y_true = [sample["target"] for sample in test_data]

    valid_models = ["xgboost", "neural-net", "random-forest"]
    if model_type not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type. Must be one of {valid_models}",
        )

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

    return {
        "model_type": model_type,
        "model_path": model_path,
        "metrics": metrics
    }
