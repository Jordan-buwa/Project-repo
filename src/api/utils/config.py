import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APIConfig:
    """Centralized configuration manager for API endpoints."""
    
    def __init__(self):
        self.repo_root = Path(__file__).resolve().parent.parent.parent.parent
        self.config_dir = self.repo_root / "config"
        self._config_cache = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from environment or config files."""
        # First check environment variables
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        
        # Then check cached configs
        return self._config_cache.get(key, default)
    
    def load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Cache for future use
        self._config_cache.update(config)
        return config
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        validation_config = self.load_yaml_config("config_api_data-val.yaml")
        registered_models = validation_config.get("registered_models", {})
        return registered_models.get(model_type, {})
    
    @property
    def model_dir(self) -> str:
        """Get models directory path."""
        return self.get("MODEL_DIR", str(self.repo_root / "models"))
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory path."""
        return str(self.repo_root / "src" / "api" / "cache")
    
    @property
    def logs_dir(self) -> str:
        """Get logs directory path."""
        return str(self.repo_root / "src" / "api" / "logs")
    
    @property
    def preprocessing_artifacts_path(self) -> str:
        """Get preprocessing artifacts file path."""
        return self.get("PREPROCESSING_ARTIFACTS_PATH", 
                       str(self.repo_root / "src" / "data_pipeline" / "preprocessing_artifacts.json"))
    
    @property
    def test_data_path(self) -> str:
        """Get test data file path."""
        return self.get("TEST_DATA_PATH", str(self.repo_root / "test_input.json"))
    
    @property
    def predict_input_path(self) -> str:
        """Get predict input file path."""
        return self.get("PREDICT_INPUT_PATH", str(self.repo_root / "src" / "api" / "routers" / "predict_input.json"))

# Global configuration instance
config = APIConfig()

# Convenience functions for common configurations
def get_model_path(model_type: str) -> str:
    """Get the latest model file path for a given model type."""
    model_dir = Path(config.model_dir)
    
    # Model file extensions mapping
    model_extensions = {
        "neural-net": ".pth",
        "xgboost": ".joblib", 
        "random-forest": ".joblib"
    }
    
    ext = model_extensions.get(model_type, ".joblib")
    
    # Find latest model file
    model_files = [
        f for f in model_dir.iterdir()
        if f.is_file() and f.suffix == ext and model_type.replace("-", "") in f.name.lower()
    ]
    
    if not model_files:
        raise FileNotFoundError(f"No trained {model_type} model found in {model_dir}")
    
    latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
    return str(latest_model)

def get_allowed_model_types() -> list:
    """Get list of allowed model types."""
    return ["xgboost", "random-forest", "neural-net"]