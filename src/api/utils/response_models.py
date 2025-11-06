from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List, Union
from enum import Enum
import datetime

class ResponseStatus(str, Enum):
    """Standard response statuses."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"

class ErrorCode(str, Enum):
    """Standard error codes."""
    # Client errors (4xx)
    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    UNPROCESSABLE_ENTITY = "UNPROCESSABLE_ENTITY"
    
    # Server errors (5xx)
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    
    # Business logic errors
    VALIDATION_FAILED = "VALIDATION_FAILED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    PREPROCESSING_FAILED = "PREPROCESSING_FAILED"
    TRAINING_FAILED = "TRAINING_FAILED"
    PREDICTION_FAILED = "PREDICTION_FAILED"

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    status: ResponseStatus
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    request_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime.datetime: lambda v: v.isoformat()
        }

class SuccessResponse(APIResponse):
    """Standard success response."""
    status: ResponseStatus = ResponseStatus.SUCCESS
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ErrorResponse(APIResponse):
    """Standard error response."""
    status: ResponseStatus = ResponseStatus.ERROR
    error_code: ErrorCode
    detail: Optional[str] = None
    errors: Optional[List[Dict[str, Any]]] = None
    path: Optional[str] = None
    method: Optional[str] = None

class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field-level errors."""
    error_code: ErrorCode = ErrorCode.VALIDATION_FAILED
    field_errors: Optional[Dict[str, List[str]]] = None

# Model-specific response models
class PredictionResponse(SuccessResponse):
    """Response for prediction endpoints."""
    data: Dict[str, Any] = Field(..., example={
        "model_type": "xgboost",
        "prediction": 0.85,
        "model_path": "models/xgboost_model_2025-10-30.joblib",
        "customer_id": "CUST123",
        "preprocessing_applied": True,
        "confidence": 0.95
    })

class MetricsResponse(SuccessResponse):
    """Response for metrics endpoints."""
    data: Dict[str, Any] = Field(..., example={
        "model_type": "xgboost",
        "model_path": "models/xgboost_model_2025-10-30.joblib",
        "metrics": {
            "accuracy": 0.86,
            "f1_score": 0.84,
            "roc_auc": 0.90
        },
        "test_samples": 1000
    })

class TrainingResponse(SuccessResponse):
    """Response for training endpoints."""
    data: Dict[str, Any] = Field(..., example={
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "model_type": "xgboost",
        "status": "pending",
        "started_at": "2025-10-30T10:00:00Z"
    })

class JobStatusResponse(SuccessResponse):
    """Response for training job status endpoints."""
    data: Dict[str, Any] = Field(..., example={
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "status": "running",
        "model_type": "xgboost",
        "script_path": "src/models/train_xgboost.py",
        "started_at": "2025-10-30T10:00:00Z",
        "completed_at": None,
        "model_path": "models/xgboost_model_2025-10-30.joblib",
        "error": None,
        "logs": "Training started...",
        "sub_jobs": [
            "8b7b6a1c-4d12-4c6f-9a2f-1234567890ab",
            "e1c2d3f4-5a6b-7c8d-9e0f-0987654321cd"
        ]
    })

class IngestionResponse(SuccessResponse):
    """Response for data ingestion endpoints."""
    data: Dict[str, Any] = Field(..., example={
        "batch_id": "BATCH123",
        "records_processed": 100,
        "records_saved": 95,
        "records_failed": 5,
        "source": "api",
        "processing_time_ms": 250
    })

class ValidationResponse(SuccessResponse):
    """Response for data validation endpoints."""
    data: Dict[str, Any] = Field(..., example={
        "rows_validated": 1000,
        "columns_checked": 45,
        "issues_found": 2,
        "validation_status": "passed",
        "model_type": "xgboost",
        "schema_version": "v20251104_010520"
    })

# Standardized data format models
class FeatureData(BaseModel):
    """Standardized feature data format."""
    features: Dict[str, Union[float, int, str]]
    customer_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TestSample(BaseModel):
    """Standardized test sample format."""
    features: Dict[str, Union[float, int, str]]
    target: Union[int, float]
    sample_id: Optional[str] = None

class TestDataset(BaseModel):
    """Standardized test dataset format."""
    samples: List[TestSample]
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'TestDataset':
        """Load test dataset from JSON file."""
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both old and new formats
        if isinstance(data, list):
            samples = []
            for item in data:
                if "features" in item and "target" in item:
                    samples.append(TestSample(**item))
                else:
                    # Handle legacy format
                    features = {k: v for k, v in item.items() if k != "target"}
                    target = item.get("target", 0)
                    samples.append(TestSample(features=features, target=target))
            return cls(samples=samples)
        else:
            return cls(**data)

class PredictionData(BaseModel):
    """Standardized prediction data format."""
    model_type: str
    prediction: Union[float, int, List[float]]
    confidence: Optional[float] = None
    model_path: Optional[str] = None
    customer_id: Optional[str] = None
    preprocessing_applied: bool = True
    metadata: Optional[Dict[str, Any]] = None

class FeatureData(BaseModel):
    """Standardized feature data format."""
    features: Dict[str, Union[float, int, str]]
    customer_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None