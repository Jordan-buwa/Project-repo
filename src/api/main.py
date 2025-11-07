from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates   
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys, os
from pathlib import Path
from contextlib import asynccontextmanager


sys.path.append(str(Path(__file__).parent.parent))


from src.api.ml_models import load_all_models, clear_models, get_all_models_info
from src.api.routers import predict, train, validate, metrics, ingest
from src.api.utils.error_handlers import api_exception_handler, validation_exception_handler
from src.api.utils.config import get_allowed_model_types 


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_startup():
    try:
        logger.info("Running startup validation...")
        from src.api.utils.setup_validator import validate_api_setup
        success, errors, warnings = validate_api_setup()
        for w in warnings: logger.warning(f"Startup warning: {w}")
        for e in errors:   logger.error(f"Startup error: {e}")
        if not success:
            logger.error("Startup validation failed with critical errors")
            return False
        logger.info("Startup validation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Startup validation crashed: {str(e)}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API server...")
    if not validate_startup():
        logger.error("Startup validation failed, but continuing...")

    try:
        logger.info("Loading ML models...")
        models = load_all_models()
        logger.info(f"Loaded {len(models)} models")
        for mt, info in get_all_models_info().items():
            if info['loaded']:
                logger.info(f"  - {mt}: {info['metadata'].get('path')}")
            else:
                logger.warning(f"  - {mt}: NOT loaded")
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        logger.warning("API will start – some endpoints may be unavailable")

    yield   

    logger.info("Shutting down API server...")
    try:
        clear_models()
        logger.info("Cleared models from memory")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


app = FastAPI(
    title="Churn Prediction API",
    description="API for training and predicting customer churn using multiple ML models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# templates

templates = Jinja2Templates(directory="src/api/templates")  

app.state.allowed_models = get_allowed_model_types()   # e.g. ["xgboost","random-forest","neural-net"]
app.state.environment   = os.getenv("ENVIRONMENT", "development")


app.add_exception_handler(Exception, api_exception_handler)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

app.include_router(predict.router, tags=["predictions"])
app.include_router(train.router,   tags=["training"])
app.include_router(validate.router,tags=["Data Validation"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(ingest.router,  tags=["Data ingestion"])

@app.get("/", response_class=HTMLResponse)
async def ui_root(request: Request):
    """Home page – uses index.html"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ingest", response_class=HTMLResponse)
async def ui_ingest(request: Request):
    return templates.TemplateResponse("ingest.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def ui_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/train", response_class=HTMLResponse)
async def ui_train(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.get("/metrics", response_class=HTMLResponse)
async def ui_metrics(request: Request):
    return templates.TemplateResponse("metrics.html", {"request": request})

@app.get("/data_view", response_class=HTMLResponse)
async def ui_data_view(request: Request):
    return templates.TemplateResponse("data_view.html", {"request": request})


@app.get("/health-ui", response_class=HTMLResponse)
async def health_ui(request: Request):
    """Human-readable health page (uses health.html)"""
    health_data = await health_check()                 
    return templates.TemplateResponse(
        "health.html",
        {"request": request, "data": health_data}
    )

@app.get("/health")
async def health_check():
    models_info = get_all_models_info()
    loaded = [mt for mt, info in models_info.items() if info['loaded']]
    return {
        "status": "healthy",
        "models_loaded": len(loaded),
        "models": models_info,
        "environment": os.getenv('ENVIRONMENT', 'development'),
        "storage_account": os.getenv('AZURE_STORAGE_ACCOUNT_NAME'),
        "version": "2.0.0"
    }

@app.get("/models")
async def get_models_status():
    return {"models": get_all_models_info()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )