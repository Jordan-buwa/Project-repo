from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional, List
import logging
import traceback
import uuid

logger = logging.getLogger(__name__)
try:
    from .response_models import ErrorResponse, ValidationErrorResponse, ErrorCode, ResponseStatus
except ImportError:
    # Fallback minimal definitions
    from enum import Enum
    
    class ResponseStatus(str, Enum):
        SUCCESS = "success"
        ERROR = "error"
        WARNING = "warning"
        PENDING = "pending"
    
    class ErrorCode(str, Enum):
        BAD_REQUEST = "BAD_REQUEST"
        UNAUTHORIZED = "UNAUTHORIZED"
        FORBIDDEN = "FORBIDDEN"
        NOT_FOUND = "NOT_FOUND"
        INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
        VALIDATION_FAILED = "VALIDATION_FAILED"
        MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
        DATA_NOT_FOUND = "DATA_NOT_FOUND"
        PREPROCESSING_FAILED = "PREPROCESSING_FAILED"
        TRAINING_FAILED = "TRAINING_FAILED"
        PREDICTION_FAILED = "PREDICTION_FAILED"

    # Minimal Pydantic models for error handling
    from pydantic import BaseModel
    from datetime import datetime
    
    class APIResponse(BaseModel):
        status: ResponseStatus
        message: str
        timestamp: str = None
        
        def __init__(self, **data):
            if 'timestamp' not in data:
                data['timestamp'] = datetime.utcnow().isoformat()
            super().__init__(**data)
    
    class ErrorResponse(APIResponse):
        status: ResponseStatus = ResponseStatus.ERROR
        error_code: ErrorCode
        detail: Optional[str] = None
    
    class ValidationErrorResponse(ErrorResponse):
        error_code: ErrorCode = ErrorCode.VALIDATION_FAILED
        field_errors: Optional[Dict[str, List[str]]] = None

class APIError(Exception):
    """Base exception for API errors."""
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: Optional[str] = None,
        errors: Optional[List[Dict[str, Any]]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.detail = detail
        self.errors = errors or []
        super().__init__(self.message)

class ValidationError(APIError):
    """Validation-specific error."""
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, List[str]]] = None,
        detail: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_FAILED,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )
        self.field_errors = field_errors or {}

class ModelNotFoundError(APIError):
    """Model not found error."""
    def __init__(self, model_type: str, detail: Optional[str] = None):
        super().__init__(
            message=f"Model '{model_type}' not found",
            error_code=ErrorCode.MODEL_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

class DataNotFoundError(APIError):
    """Data not found error."""
    def __init__(self, data_source: str, detail: Optional[str] = None):
        super().__init__(
            message=f"Data from '{data_source}' not found",
            error_code=ErrorCode.DATA_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

class PreprocessingError(APIError):
    """Data preprocessing error."""
    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.PREPROCESSING_FAILED,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

class TrainingError(APIError):
    """Model training error."""
    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.TRAINING_FAILED,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

class PredictionError(APIError):
    """Prediction error."""
    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.PREDICTION_FAILED,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

def create_error_response(
    exception: Exception,
    request: Request,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Create standardized error response from exception."""
    if not request_id:
        request_id = str(uuid.uuid4())
    
    # Handle APIError instances
    if isinstance(exception, APIError):
        return ErrorResponse(
            message=exception.message,
            error_code=exception.error_code,
            detail=exception.detail,
            errors=exception.errors,
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )
    
    # Handle FastAPI HTTPException
    elif isinstance(exception, HTTPException):
        error_code = _map_status_code_to_error_code(exception.status_code)
        return ErrorResponse(
            message=exception.detail or "An error occurred",
            error_code=error_code,
            status_code=exception.status_code,
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )
    
    # Handle validation errors
    elif hasattr(exception, 'errors') and callable(getattr(exception, 'errors', None)):
        field_errors = {}
        for error in exception.errors():
            field = error.get('loc', ['unknown'])[-1]
            msg = error.get('msg', 'Validation error')
            if field not in field_errors:
                field_errors[field] = []
            field_errors[field].append(msg)
        
        return ValidationErrorResponse(
            message="Validation failed",
            field_errors=field_errors,
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )
    
    # Handle generic exceptions
    else:
        logger.error(f"Unhandled exception: {str(exception)}\n{traceback.format_exc()}")
        return ErrorResponse(
            message="An internal server error occurred",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            detail=str(exception) if __debug__ else None,
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )

def _map_status_code_to_error_code(status_code: int) -> ErrorCode:
    """Map HTTP status code to error code."""
    status_code_map = {
        400: ErrorCode.BAD_REQUEST,
        401: ErrorCode.UNAUTHORIZED,
        403: ErrorCode.FORBIDDEN,
        404: ErrorCode.NOT_FOUND,
        409: ErrorCode.CONFLICT,
        422: ErrorCode.UNPROCESSABLE_ENTITY,
        500: ErrorCode.INTERNAL_SERVER_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
        504: ErrorCode.TIMEOUT,
    }
    return status_code_map.get(status_code, ErrorCode.INTERNAL_SERVER_ERROR)

async def api_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for API errors."""
    request_id = str(uuid.uuid4())
    
    # Log the error
    logger.error(f"Request {request_id} failed: {str(exc)}\n{traceback.format_exc()}")
    
    # Create error response
    error_response = create_error_response(exc, request, request_id)
    
    # Determine status code
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, APIError):
        status_code = exc.status_code
    elif isinstance(exc, HTTPException):
        status_code = exc.status_code
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.dict(exclude_none=True)
    )

async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for validation errors."""
    request_id = str(uuid.uuid4())
    
    # Log the validation error
    logger.warning(f"Validation failed for request {request_id}: {str(exc)}")
    
    # Create validation error response
    error_response = create_error_response(exc, request, request_id)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.dict(exclude_none=True)
    )

def handle_model_error(model_type: str, error: Exception) -> None:
    """Handle model-related errors consistently."""
    logger.error(f"Model error for {model_type}: {str(error)}")
    
    if "not found" in str(error).lower() or "does not exist" in str(error).lower():
        raise ModelNotFoundError(model_type, detail=str(error))
    else:
        raise PredictionError(f"Failed to process model {model_type}: {str(error)}")

def handle_data_error(data_source: str, error: Exception) -> None:
    """Handle data-related errors consistently."""
    logger.error(f"Data error for {data_source}: {str(error)}")
    
    if "not found" in str(error).lower() or "does not exist" in str(error).lower():
        raise DataNotFoundError(data_source, detail=str(error))
    else:
        raise PreprocessingError(f"Failed to process data from {data_source}: {str(error)}")

def handle_training_error(model_type: str, error: Exception) -> None:
    """Handle training-related errors consistently."""
    logger.error(f"Training error for {model_type}: {str(error)}")
    raise TrainingError(f"Failed to train model {model_type}: {str(error)}")

# Utility functions for common error scenarios
def raise_if_model_not_found(model_path: str, model_type: str) -> None:
    """Raise ModelNotFoundError if model path is invalid."""
    import os
    if not model_path or not os.path.exists(model_path):
        raise ModelNotFoundError(model_type, f"Model file not found: {model_path}")

def raise_if_data_empty(data: Any, data_source: str) -> None:
    """Raise DataNotFoundError if data is empty."""
    if not data or (hasattr(data, '__len__') and len(data) == 0):
        raise DataNotFoundError(data_source, "No data available for processing")