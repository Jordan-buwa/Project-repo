import os

# Can store default model versions or stages for API awareness
MODEL_STAGES = {
    "random_forest": "Production",
    "xgboost": "Production",
    "neural_network": "Production"
}

def get_model_stage(model_type: str):
    return MODEL_STAGES.get(model_type, "Unknown")
