# src/api/train_api.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
import subprocess
import os
import uuid
from datetime import datetime
from typing import Dict

app = FastAPI(title="Model Training API")

# Simple in-memory job registry
jobs: Dict[str, Dict] = {}

def run_training_script(script_path: str, job_id: str):
    """Run the training script and update job status."""
    try:
        jobs[job_id]["status"] = "running"
        subprocess.run(["python", script_path], check=True)
        # Save path to the latest model file (assumes script saves with timestamp)
        latest_model = max(
            [f for f in os.listdir("models") if f.endswith(".pth") or f.endswith(".joblib")],
            key=lambda f: os.path.getmtime(os.path.join("models", f))
        )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["model_path"] = os.path.join("models", latest_model)
    except subprocess.CalledProcessError as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

def validate_script(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Script not found: {path}")
    return path

def start_job(script_path: str):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "started_at": datetime.utcnow().isoformat()}
    return job_id

# ----------------- Endpoints -----------------
@app.post("/train/neural-net")
def train_nn(background_tasks: BackgroundTasks):
    script = validate_script("src/models/churn_nn.py")
    job_id = start_job(script)
    background_tasks.add_task(run_training_script, script, job_id)
    return {"job_id": job_id, "status": "started", "model_type": "neural-net"}

@app.post("/train/xgboost")
def train_xgb(background_tasks: BackgroundTasks):
    script = validate_script("src/models/train_xgboost.py")
    job_id = start_job(script)
    background_tasks.add_task(run_training_script, script, job_id)
    return {"job_id": job_id, "status": "started", "model_type": "xgboost"}

@app.post("/train/random-forest")
def train_rf(background_tasks: BackgroundTasks):
    script = validate_script("src/models/train_RandomForest.py")
    job_id = start_job(script)
    background_tasks.add_task(run_training_script, script, job_id)
    return {"job_id": job_id, "status": "started", "model_type": "random-forest"}

@app.get("/train/status/{job_id}")
def job_status(job_id: str):
    """Query job status and model path (if completed)"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return jobs[job_id]
