from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel
import subprocess
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)

# Training job registry (use Redis or database in production)
training_jobs: Dict[str, Dict] = {}

class TrainingRequest(BaseModel):
    model_type: str  # "neural-net", "xgboost", "random-forest", "all"
    retrain: bool = False
    use_cv: bool = True
    hyperparameters: Optional[Dict] = None

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    model_type: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    model_type: str
    started_at: str
    completed_at: Optional[str] = None
    model_path: Optional[str] = None
    error: Optional[str] = None
    logs: Optional[str] = None

def validate_training_script(script_path: str) -> str:
    """Validate that the training script exists."""
    path = Path(script_path)
    if not path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Training script not found: {script_path}"
        )
    return str(path)

def create_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid.uuid4())

def register_job(job_id: str, model_type: str, script_path: str):
    """Register a new training job."""
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "model_type": model_type,
        "script_path": script_path,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "model_path": None,
        "error": None,
        "logs": ""
    }
    return job_id

def run_training_script(script_path: str, job_id: str, model_type: str):
    """Run the training script and update job status."""
    try:
        # Update job status to running
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Starting training job {job_id} for {model_type}")
        
        # Run the training script
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Update job status on success
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        training_jobs[job_id]["logs"] = result.stdout
        
        # Find the latest model file
        model_path = find_latest_model_file(model_type)
        if model_path:
            training_jobs[job_id]["model_path"] = model_path
            logger.info(f"Training completed for job {job_id}. Model saved: {model_path}")
        else:
            logger.warning(f"Training completed for job {job_id} but no model file found")
            
    except subprocess.CalledProcessError as e:
        # Update job status on failure
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        training_jobs[job_id]["error"] = f"Script execution failed: {e.stderr}"
        training_jobs[job_id]["logs"] = e.stdout + "\n" + e.stderr
        logger.error(f"Training failed for job {job_id}: {e.stderr}")
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        training_jobs[job_id]["error"] = str(e)
        logger.error(f"Unexpected error in training job {job_id}: {str(e)}")

def find_latest_model_file(model_type: str) -> Optional[str]:
    """Find the latest model file based on model type."""
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    model_patterns = {
        "neural-net": ["*.h5", "*.pth", "*.pt"],
        "xgboost": ["*xgboost*", "*.pkl", "*.joblib"],
        "random-forest": ["*random_forest*", "*.pkl", "*.joblib"]
    }
    
    patterns = model_patterns.get(model_type, ["*.pkl", "*.joblib", "*.h5"])
    
    latest_file = None
    latest_time = 0
    
    for pattern in patterns:
        for model_file in models_dir.glob(pattern):
            file_time = model_file.stat().st_mtime
            if file_time > latest_time:
                latest_time = file_time
                latest_file = str(model_file)
    
    return latest_file

def get_script_path(model_type: str) -> str:
    """Get the script path for the specified model type."""
    script_map = {
        "neural-net": "src/models/churn_nn.py",
        "xgboost": "src/models/train_xgboost.py", 
        "random-forest": "src/models/train_RandomForest.py"
    }
    
    if model_type not in script_map:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported model type: {model_type}. Supported types: {list(script_map.keys())}"
        )
    
    return script_map[model_type]

@router.post("/train/{model_type}", response_model=TrainingResponse)
async def train_model(
    model_type: str,
    background_tasks: BackgroundTasks,
    request: Optional[TrainingRequest] = None
):
    """
    Start training for a specific model type.
    
    Args:
        model_type: Type of model to train ("neural-net", "xgboost", "random-forest")
        background_tasks: FastAPI background tasks
        request: Optional training configuration
    """
    try:
        # Validate model type and get script path
        script_path = get_script_path(model_type)
        validated_script = validate_training_script(script_path)
        
        # Create and register job
        job_id = create_job_id()
        register_job(job_id, model_type, validated_script)
        
        # Start training in background
        background_tasks.add_task(
            run_training_script, 
            validated_script, 
            job_id, 
            model_type
        )
        
        logger.info(f"Started training job {job_id} for {model_type}")
        
        return TrainingResponse(
            job_id=job_id,
            status="started",
            message=f"Training initiated for {model_type}",
            model_type=model_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train", response_model=TrainingResponse)
async def train_model_with_config(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start training with full configuration.
    
    Args:
        request: Training configuration including model type and parameters
        background_tasks: FastAPI background tasks
    """
    try:
        if request.model_type == "all":
            # Start training for all model types
            job_id = create_job_id()
            training_jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "model_type": "all",
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": None,
                "sub_jobs": []
            }
            
            # Start individual training jobs for each model type
            model_types = ["neural-net", "xgboost", "random-forest"]
            for model_type in model_types:
                sub_job_id = await start_single_training(
                    model_type, background_tasks, request
                )
                training_jobs[job_id]["sub_jobs"].append(sub_job_id)
            
            return TrainingResponse(
                job_id=job_id,
                status="started", 
                message="Training initiated for all model types",
                model_type="all"
            )
        else:
            # Single model training
            job_id = await start_single_training(
                request.model_type, background_tasks, request
            )
            
            return TrainingResponse(
                job_id=job_id,
                status="started",
                message=f"Training initiated for {request.model_type}",
                model_type=request.model_type
            )
            
    except Exception as e:
        logger.error(f"Error in train with config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def start_single_training(
    model_type: str, 
    background_tasks: BackgroundTasks,
    request: TrainingRequest
) -> str:
    """Start training for a single model type."""
    script_path = get_script_path(model_type)
    validated_script = validate_training_script(script_path)
    
    job_id = create_job_id()
    register_job(job_id, model_type, validated_script)
    
    # Add hyperparameters to job info if provided
    if request.hyperparameters:
        training_jobs[job_id]["hyperparameters"] = request.hyperparameters
    
    background_tasks.add_task(
        run_training_script, 
        validated_script, 
        job_id, 
        model_type
    )
    
    return job_id

@router.get("/train/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a training job.
    
    Args:
        job_id: The ID of the training job to check
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    job_info = training_jobs[job_id]
    
    # For "all" jobs, aggregate status from sub-jobs
    if job_info["model_type"] == "all" and "sub_jobs" in job_info:
        sub_jobs = job_info["sub_jobs"]
        if all(training_jobs.get(sub_id, {}).get("status") == "completed" for sub_id in sub_jobs):
            job_info["status"] = "completed"
        elif any(training_jobs.get(sub_id, {}).get("status") == "failed" for sub_id in sub_jobs):
            job_info["status"] = "failed"
        elif any(training_jobs.get(sub_id, {}).get("status") == "running" for sub_id in sub_jobs):
            job_info["status"] = "running"
    
    return JobStatusResponse(**job_info)

@router.get("/train/jobs")
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """
    List all training jobs with optional filtering.
    
    Args:
        limit: Maximum number of jobs to return
        status: Filter by job status ("pending", "running", "completed", "failed")
    """
    jobs_list = list(training_jobs.values())
    
    if status:
        jobs_list = [job for job in jobs_list if job.get("status") == status]
    
    # Sort by start time (newest first)
    jobs_list.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    
    return {
        "jobs": jobs_list[:limit],
        "total_count": len(jobs_list),
        "filtered_count": len(jobs_list[:limit])
    }

@router.delete("/train/job/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a training job (if possible).
    
    Note: This is a basic implementation. For full cancellation support,
    you would need to implement process management.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    job = training_jobs[job_id]
    
    if job["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel job with status: {job['status']}"
        )
    
    # Update status to cancelled
    job["status"] = "cancelled"
    job["completed_at"] = datetime.utcnow().isoformat()
    job["error"] = "Job was cancelled by user"
    
    logger.info(f"Cancelled training job {job_id}")
    
    return {"message": f"Job {job_id} cancelled successfully", "job_id": job_id}

@router.get("/train/models/available")
async def get_available_models():
    """Get list of available trained models in the models directory."""
    models_dir = Path("models")
    available_models = {}
    
    if models_dir.exists():
        for model_file in models_dir.iterdir():
            if model_file.is_file():
                file_type = model_file.suffix.lower()
                model_type = "unknown"
                
                if "neural" in model_file.name.lower() or "nn" in model_file.name.lower():
                    model_type = "neural-net"
                elif "xgboost" in model_file.name.lower() or "xgb" in model_file.name.lower():
                    model_type = "xgboost" 
                elif "random" in model_file.name.lower() or "rf" in model_file.name.lower():
                    model_type = "random-forest"
                
                available_models[model_file.name] = {
                    "path": str(model_file),
                    "type": model_type,
                    "size": model_file.stat().st_size,
                    "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                }
    
    return {
        "available_models": available_models,
        "models_directory": str(models_dir.absolute())
    }