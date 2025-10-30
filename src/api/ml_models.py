import pickle
import torch

ml_models = {} 

def load_pickle_model(path: str):
    """Load a pickle model"""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def load_torch_model(path: str, model_class=None):
    """Load a PyTorch model"""
    if model_class is not None:
        model_class.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model_class.eval()
        return model_class
    else:
        model = torch.load(path, map_location=torch.device('cpu'))
        model.eval()
        return model

def load_all_models():
    """Load all ML models at startup"""
    ml_models['xgboost'] = load_pickle_model("./models/xgboost.pkl")
    ml_models['random_forest'] = load_pickle_model("./models/random_forest.pkl")
    ml_models['Neuralnet'] = load_torch_model("./models/model.pth")
    return ml_models

def clear_models():
    """Clear all models from memory"""
    ml_models.clear()