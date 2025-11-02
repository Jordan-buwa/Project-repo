from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.ml_models import load_all_models, clear_models, get_all_models_info
from api.routers import predict, train, validate, metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("Starting API server...")
    
    try:
        # Load all models at startup
        logger.info("Loading ML models...")
        models = load_all_models()
        logger.info(f"Loaded {len(models)} models successfully")
        
        # Log model info
        models_info = get_all_models_info()
        for model_type, info in models_info.items():
            if info['loaded']:
                logger.info(f"  - {model_type}: Loaded from {info['metadata'].get('path')}")
            else:
                logger.warning(f"  - {model_type}: Not loaded")
    
    except Exception as e:
        logger.error(f"Error loading models at startup: {str(e)}")
        logger.warning("API will start but some endpoints may not work without models")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    try:
        clear_models()
        logger.info("Cleared models from memory")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Churn Prediction API",
    description="API for training and predicting customer churn using multiple ML models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )

# Include routers
app.include_router(predict.router, prefix="/api/v1", tags=["predictions"])
app.include_router(train.router, prefix="/api/v1", tags=["training"])
#app.include_router(validate.router, prefix="/api/v1", tags=["validation"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "models": "/api/v1/models"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_info = get_all_models_info()
    
    loaded_models = [
        model_type for model_type, info in models_info.items() 
        if info['loaded']
    ]
    
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "models": models_info
    }

@app.get("/api/v1/models")
async def get_models_status():
    """Get status of all loaded models"""
    return {
        "models": get_all_models_info()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )