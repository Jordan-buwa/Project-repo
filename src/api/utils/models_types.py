from enum import Enum

class ModelType(str, Enum):
    """Standardized model type definitions"""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest" 
    NEURAL_NET = "neural_net"
    
    @classmethod
    def get_all_types(cls):
        return [model_type.value for model_type in cls]

# Convenience functions
def validate_model_type(model_type: str) -> bool:
    """Validate if model type is supported"""
    return model_type in ModelType.get_all_types()

def normalize_model_type(model_type: str) -> str:
    """Convert hyphens to underscores for consistency"""
    return model_type.replace("-", "_")