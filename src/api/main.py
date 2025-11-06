# src/api/main.py
from fastapi import FastAPI
from .model_versions import router as model_versions_router
from .data_validation import router as data_validation_router

app = FastAPI(
    title="Model Versioning and Data Validation API",
    description="API for managing MLflow model versions and validating incoming datasets.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """
    Simple health check endpoint to confirm the API is running.
    """
    return {"status": "ok", "message": "API is running smoothly"}


# Register routers
app.include_router(model_versions_router)
app.include_router(data_validation_router)
