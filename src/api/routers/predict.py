from fastapi import FastAPI, HTTPException, APIRouter
import os
import joblib
import json

router = APIRouter()
app = FastAPI(title="Model Prediction API")


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


@app.post("/predict/{model_type}")
def predict(model_type: str):
    # Load features directly from JSON
    input_path = os.path.join(os.path.dirname(__file__), "predict_input.json")
    if not os.path.exists(input_path):
        raise HTTPException(
            status_code=404, detail="predict_input.json not found")

    with open(input_path, "r") as f:
        data = json.load(f)

    # single sample as 2D list
    features_list = [list(data["features"].values())]

    valid_models = ["xgboost", "neural-net", "random-forest"]
    if model_type not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type. Must be one of {valid_models}",
        )

    model_path = get_latest_model(model_type)
    model = load_model(model_path, model_type)

    if model_type == "neural-net":
        import torch
        X = torch.tensor(features_list, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(X).numpy()
    else:
        prediction = model.predict(features_list)

    return {
        "model_type": model_type,
        "prediction": prediction.tolist(),
        "model_path": model_path,
    }
