import pickle
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from .utils.models_types import ModelType, validate_model_type, normalize_model_type


logger = logging.getLogger(__name__)

# Global model registry
ml_models: Dict[str, Any] = {}
model_metadata: Dict[str, Dict] = {}

class ModelLoadError(Exception):
    """Custom exception for model loading errors"""
    pass

from pathlib import Path
from typing import Optional

def get_model_path(model_type: str) -> Path:
    """
    Get the most recent model file from the 'models/' directory that matches
    the identifier for the given model_type.
    """
    # Normalize model type first
    normalized_type = normalize_model_type(model_type)
    
    if not validate_model_type(normalized_type):
        raise ValueError(f"Unknown model type: {model_type}. Supported: {ModelType.get_all_types()}")
    
    models_dir = Path("models/")
    if not models_dir.exists() or not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    identifiers = {
        ModelType.XGBOOST: {"xgboost", "xgb"},
        ModelType.RANDOM_FOREST: {"random_forest", "random-forest", "rf", ".joblib"},
        ModelType.NEURAL_NET: {"neural", "nn", ".pth", ".h5", ".pt"},
    }

    patterns = identifiers[normalized_type]
    
    candidates: list[tuple[float, Path]] = []
    
    for file_path in models_dir.iterdir():
        if not file_path.is_file():
            continue
        
        name_lower = file_path.name.lower()
        suffix_lower = file_path.suffix.lower()
        
        matches = any(
            pat.lstrip(".") in name_lower or  # substring match
            (pat.startswith(".") and pat == suffix_lower)  # exact suffix match
            for pat in patterns
        )
        
        if matches:
            mtime = file_path.stat().st_mtime
            candidates.append((mtime, file_path))
    
    if not candidates:
        raise FileNotFoundError(f"No model file found for type '{model_type}' in {models_dir}")
    
    # Return the most recently modified file
    latest_path = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
    return latest_path

def load_pickle_model(path: Path) -> Any:
    """Load a pickle model with error handling"""
    try:
        if not path.exists():
            raise ModelLoadError(f"Model file not found: {path}")
        
        with open(path, "rb") as f:
            model = pickle.load(f)
        
        logger.info(f"Successfully loaded pickle model from {path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading pickle model from {path}: {str(e)}")
        raise ModelLoadError(f"Failed to load pickle model: {str(e)}")

def load_torch_model(path: Path, model_class=None) -> torch.nn.Module:
    """Load a PyTorch model with error handling"""
    try:
        if not path.exists():
            raise ModelLoadError(f"Model file not found: {path}")
        
        if model_class is not None:
            model_class.load_state_dict(
                torch.load(path, map_location=torch.device('cpu'))
            )
            model_class.eval()
            model = model_class
        else:
            model = torch.load(path, map_location=torch.device('cpu'))
            model.eval()
        
        logger.info(f"Successfully loaded PyTorch model from {path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading PyTorch model from {path}: {str(e)}")
        raise ModelLoadError(f"Failed to load PyTorch model: {str(e)}")

def load_single_model(model_type: str, force_reload: bool = False) -> Optional[Any]:
    """
    Load a single model by type
    """
    try:
        normalized_type = normalize_model_type(model_type)
        
        if normalized_type in ml_models and not force_reload:
            logger.info(f"Model {normalized_type} already loaded, using cached version")
            return ml_models[normalized_type]
        
        path = get_model_path(normalized_type)
        
        # Load based on model type
        if normalized_type == ModelType.NEURAL_NET:
            model = load_torch_model(path)
        else:
            model = load_pickle_model(path)
        
        # Store model and metadata
        ml_models[normalized_type] = model
        model_metadata[normalized_type] = {
            'loaded_at': datetime.utcnow().isoformat(),
            'path': str(path),
            'size': path.stat().st_size,
            'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        }
        
        logger.info(f"Successfully loaded model: {model_type}")
        return model
    
    except ModelLoadError as e:
        logger.warning(f"Could not load model {model_type}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading model {model_type}: {str(e)}")
        return None

def load_all_models(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load all ML models at startup
    """
    model_types = ModelType.get_all_types()    
    loaded_count = 0
    failed_models = []
    
    for model_type in model_types:
        try:
            if load_single_model(model_type, force_reload=force_reload):
                loaded_count += 1
            else:
                failed_models.append(model_type)
        except Exception as e:
            logger.error(f"Failed to load {model_type}: {str(e)}")
            failed_models.append(model_type)
    
    logger.info(f"Loaded {loaded_count}/{len(model_types)} models")
    
    if failed_models:
        logger.warning(f"Failed to load models: {', '.join(failed_models)}")
    
    return ml_models

def reload_model(model_type: str) -> bool:
    """
    Reload a specific model 
    """
    try:
        # Remove old model from memory
        if model_type in ml_models:
            del ml_models[model_type]
            logger.info(f"Removed old {model_type} model from memory")
        
        # Load new model
        model = load_single_model(model_type, force_reload=True)
        
        if model is not None:
            logger.info(f"Successfully reloaded model: {model_type}")
            return True
        else:
            logger.warning(f"Failed to reload model: {model_type}")
            return False
    
    except Exception as e:
        logger.error(f"Error reloading model {model_type}: {str(e)}")
        return False

def get_model(model_type: str) -> Optional[Any]:
    """
    Get a loaded model by type
    """
    return ml_models.get(model_type)

def is_model_loaded(model_type: str) -> bool:
    """Check if a model is currently loaded"""
    return model_type in ml_models

def get_model_info(model_type: str) -> Optional[Dict]:
    """Get metadata about a loaded model"""
    return model_metadata.get(model_type)

def get_all_models_info() -> Dict[str, Dict]:
    """Get metadata about all loaded models"""
    return {
        model_type: {
            'loaded': is_model_loaded(model_type),
            'metadata': model_metadata.get(model_type, {})
        }
        for model_type in ModelType.get_all_types()
    }

def clear_models():
    """Clear all models from memory"""
    ml_models.clear()
    model_metadata.clear()
    logger.info("Cleared all models from memory")

def check_models_availability() -> Dict[str, bool]:
    """Check which model files are available on disk"""
    availability = {}
    
    for model_type in ['xgboost', 'random_forest', 'neural_net']:
        try:
            path = get_model_path(model_type)
            availability[model_type] = path.exists()
        except Exception:
            availability[model_type] = False
    
    return availability