from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, status
from pydantic import BaseModel
import subprocess
from dotenv import load_dotenv
import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from src.api.utils.config import APIConfig, get_allowed_model_types
from src.api.utils.response_models import TrainingResponse, JobStatusResponse
from src.api.utils.error_handlers import TrainingError, handle_training_error

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent 
sys.path.append(str(REPO_ROOT))  

os.environ["REPO_ROOT"] = str(REPO_ROOT)
os.environ["PYTHONPATH"] = str(REPO_ROOT)

router = APIRouter(prefix="/train")
logger = logging.getLogger(__name__)

# Initialize configuration
config = APIConfig()

# Setup logging with centralized config
log_path = config.logs_dir
os.makedirs(log_path, exist_ok=True)
log_file = os.path.join(log_path, 'training_jobs.log')
logging.basicConfig(
    filename=log_file,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)

# Training job registry (use Redis or database in production)
training_jobs: Dict[str, Dict] = {}

class TrainingRequest(BaseModel):
    model_type: str  # "neural-net", "xgboost", "random-forest", "all"
    retrain: bool = False
    use_cv: bool = True
    hyperparameters: Optional[Dict] = None

def validate_training_script(script_path: str) -> str:
    """Validate that the training script exists (resolve relative to repo root)."""
    repo_root = config.repo_root
    path = Path(script_path)
    if not path.is_absolute():
        candidate = repo_root / path
    else:
        candidate = path
    if not candidate.exists():
        raise TrainingError(
            status_code=status.HTTP_404_NOT_FOUND, 
            message=f"Training script not found: {candidate}"
        )
    return str(candidate)

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
        
        # Ensure subprocess runs with repository root on PYTHONPATH so `import src` works
        repo_root = config.repo_root
        
        # If the script lives under src/, prefer running it as a module to preserve package imports
        rel_path = Path(script_path).relative_to(repo_root) if Path(script_path).is_absolute() else Path(script_path)
        run_cmd = None
        if str(rel_path).startswith("src" + os.sep) or str(rel_path).startswith("src/"):
            # convert src/models/train_xgboost.py -> src.models.train_xgboost
            module = str(rel_path).replace(os.sep, ".")
            if module.endswith(".py"):
                module = module[:-3]
            run_cmd = [sys.executable, "-m", module]
        else:
            run_cmd = [sys.executable, script_path]
        
        logger.info(f"Running command: {' '.join(run_cmd)} (cwd={repo_root})")
        
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
            env=os.environ
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
        training_jobs[job_id]["logs"] = (e.stdout or "") + "\n" + (e.stderr or "")
        logger.error(f"Training failed for job {job_id}: {e.stderr}\nCmd: {getattr(e, 'cmd', None)}")
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        training_jobs[job_id]["error"] = str(e)
        logger.error(f"Unexpected error in training job {job_id}: {str(e)}")

def find_latest_model_file(model_type: str) -> Optional[str]:
    """Find the latest model file based on model type."""
    models_dir = Path(config.models_dir)
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
    allowed_types = get_allowed_model_types()
    
    script_map = {
        "neural-net": "src/models/train_nn.py",
        "xgboost": "src/models/train_xgb.py", 
        "random-forest": "src/models/train_rf.py"
    }
    
    if model_type not in allowed_types:
        raise TrainingError(
            status_code=status.HTTP_400_BAD_REQUEST, 
            message=f"Unsupported model type: {model_type}. Supported types: {allowed_types}"
        )
    
    return script_map[model_type]

@router.post("/{model_type}", response_model=TrainingResponse)
async def train_model(
    model_type: str,
    background_tasks: BackgroundTasks,
    request: Optional[TrainingRequest] = None
):
    """
    Start training for a specific model type.
    
    Args:
        model_type: The type of model to train (neural-net, xgboost, random-forest)
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
            status="success",
            message=f"Training initiated for {model_type}",
            data={
                "job_id": job_id,
                "model_type": model_type,
                "status": "started"
            }
        )
        
    except TrainingError:
        raise
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        handle_training_error(model_type, e)

@router.post("/{job_id}", response_model=TrainingResponse)
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

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a training job.
    
    Args:
        job_id: The ID of the training job to check
    """
    if job_id not in training_jobs:
        raise TrainingError(
            status_code=status.HTTP_404_NOT_FOUND, 
            message="Job ID not found"
        )
    
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
    
    return JobStatusResponse(
        status="success",
        message="Job status retrieved successfully",
        data=job_info
    )

@router.get("/jobs")
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

@router.delete("/job/cancel/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a training job (if possible).
    
    Note: This is a basic implementation. For full cancellation support,
    you would need to implement process management.
    """
    if job_id not in training_jobs:
        raise TrainingError(
            status_code=status.HTTP_404_NOT_FOUND, 
            message="Job ID not found"
        )
    
    job = training_jobs[job_id]
    
    if job["status"] in ["completed", "failed"]:
        raise TrainingError(
            status_code=status.HTTP_400_BAD_REQUEST, 
            message=f"Cannot cancel job with status: {job['status']}"
        )
    
    # Update status to cancelled
    job["status"] = "cancelled"
    job["completed_at"] = datetime.utcnow().isoformat()
    job["error"] = "Job was cancelled by user"
    
    logger.info(f"Cancelled training job {job_id}")
    
    return {
        "status": "success",
        "message": f"Job {job_id} cancelled successfully", 
        "data": {"job_id": job_id}
    }
@router.get("/models/available")
async def get_available_models():
    """Get list of available trained models in the models directory."""
    models_dir = Path(config.model_dir)
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