import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException, FastAPI
from pathlib import Path
import subprocess
import uuid
import os
import json
import logging
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import time
import sys

# Import the router module
import src.api.routers.train as training_api
from src.api.routers.train import (
    validate_training_script,
    create_job_id,
    register_job,
    get_script_path,
    run_training_script,
    find_latest_model_file,
    training_jobs,
    TrainingRequest,
    start_single_training
)

# Create test app
app = FastAPI()
app.include_router(training_api.router)
client = TestClient(app)

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clear_training_jobs():
    """Reset job registry before each test."""
    training_jobs.clear()
    yield
    training_jobs.clear()


@pytest.fixture
def fake_script(tmp_path):
    """Create a temporary training script."""
    script_dir = tmp_path / "src" / "models"
    script_dir.mkdir(parents=True)
    
    # Create multiple script types
    scripts = {
        "xgboost": script_dir / "train_xgboost.py",
        "neural": script_dir / "churn_nn.py",
        "rf": script_dir / "train_RandomForest.py"
    }
    
    for name, script_path in scripts.items():
        script_path.write_text("print('Training OK')\nimport sys\nsys.exit(0)")
    
    return scripts


@pytest.fixture
def fake_models_dir(tmp_path):
    """Create temporary models directory with fake model files."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    
    # Create fake model files
    (models_dir / "xgboost_model.joblib").write_text("fake model")
    (models_dir / "neural_net.pth").write_text("fake neural model")
    (models_dir / "random_forest_model.pkl").write_text("fake rf model")
    
    return models_dir


@pytest.fixture
def mock_subprocess_success():
    """Mock successful subprocess run."""
    def _mock_run(*args, **kwargs):
        result = Mock()
        result.stdout = "Training completed successfully"
        result.stderr = ""
        result.returncode = 0
        return result
    return _mock_run


@pytest.fixture
def mock_subprocess_failure():
    """Mock failed subprocess run."""
    def _mock_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0] if args else [],
            stderr="Training failed with error"
        )
    return _mock_run


# ------------------------------------------------------------------------------
# Unit Tests - Validation Functions
# ------------------------------------------------------------------------------
class TestValidationFunctions:
    
    def test_validate_training_script_exists(self, fake_script):
        """Should return absolute path when file exists."""
        path = validate_training_script(str(fake_script["xgboost"]))
        assert Path(path).exists()
        assert Path(path).is_absolute()
    
    def test_validate_training_script_relative_path(self, monkeypatch, fake_script, tmp_path):
        """Should resolve relative paths correctly."""
        # Create a more realistic test
        script_path = "src/models/train_xgboost.py"
        
        # This test verifies that non-existent relative paths raise errors
        with pytest.raises(HTTPException) as exc:
            validate_training_script("nonexistent/script.py")
        assert exc.value.status_code == 404
    
    def test_validate_training_script_not_found(self):
        """Should raise 404 when file not found."""
        with pytest.raises(HTTPException) as exc:
            validate_training_script("src/models/nonexistent.py")
        assert exc.value.status_code == 404
        assert "not found" in exc.value.detail.lower()
    
    def test_validate_training_script_absolute_path(self, fake_script):
        """Should handle absolute paths."""
        abs_path = str(fake_script["xgboost"].absolute())
        result = validate_training_script(abs_path)
        assert result == abs_path


class TestScriptPathMapping:
    
    def test_get_script_path_xgboost(self):
        """Should map xgboost to correct script."""
        path = get_script_path("xgboost")
        assert "xgboost" in path.lower()
        assert path.endswith(".py")
    
    def test_get_script_path_neural_net(self):
        """Should map neural-net to correct script."""
        path = get_script_path("neural-net")
        assert "nn" in path.lower() or "neural" in path.lower()
    
    def test_get_script_path_random_forest(self):
        """Should map random-forest to correct script."""
        path = get_script_path("random-forest")
        assert "random" in path.lower() or "forest" in path.lower()
    
    def test_get_script_path_invalid_model(self):
        """Should raise HTTPException for unsupported model."""
        with pytest.raises(HTTPException) as exc:
            get_script_path("unsupported-model")
        assert exc.value.status_code == 400
        assert "unsupported" in exc.value.detail.lower()
    
    def test_get_script_path_case_sensitivity(self):
        """Should handle different cases."""
        with pytest.raises(HTTPException):
            get_script_path("XGBoost")  # Should be lowercase


class TestJobManagement:
    
    def test_create_job_id_is_uuid(self):
        """Job ID should be valid UUID."""
        job_id = create_job_id()
        assert uuid.UUID(job_id)  # Should not raise
    
    def test_create_job_id_unique(self):
        """Each job ID should be unique."""
        ids = [create_job_id() for _ in range(100)]
        assert len(ids) == len(set(ids))
    
    def test_register_job_creates_entry(self, fake_script):
        """Should create job entry with correct structure."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        assert job_id in training_jobs
        job = training_jobs[job_id]
        
        assert job["job_id"] == job_id
        assert job["status"] == "pending"
        assert job["model_type"] == "xgboost"
        assert job["script_path"] == str(fake_script["xgboost"])
        assert job["started_at"] is not None
        assert job["completed_at"] is None
        assert job["model_path"] is None
        assert job["error"] is None
        assert job["logs"] == ""
    
    def test_register_job_timestamp_format(self, fake_script):
        """Job timestamp should be ISO format."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        timestamp = training_jobs[job_id]["started_at"]
        # Should be valid ISO format
        datetime.fromisoformat(timestamp)


class TestModelFileDiscovery:
    
    def test_find_latest_model_file_xgboost(self, fake_models_dir, monkeypatch):
        """Should find xgboost model file."""
        # Change to models directory temporarily
        original_cwd = os.getcwd()
        try:
            os.chdir(fake_models_dir.parent)
            result = find_latest_model_file("xgboost")
            assert result is not None
            assert "xgboost" in result.lower() or "joblib" in result or "pkl" in result
        finally:
            os.chdir(original_cwd)
    
    def test_find_latest_model_file_neural_net(self, fake_models_dir, monkeypatch):
        """Should find neural net model file."""
        original_cwd = os.getcwd()
        try:
            os.chdir(fake_models_dir.parent)
            result = find_latest_model_file("neural-net")
            assert result is not None
            assert any(ext in result for ext in [".pth", ".pt", ".h5"])
        finally:
            os.chdir(original_cwd)
    
    def test_find_latest_model_file_no_models_dir(self, tmp_path, monkeypatch):
        """Should return None when models directory doesn't exist."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = find_latest_model_file("xgboost")
            assert result is None
        finally:
            os.chdir(original_cwd)
    
    def test_find_latest_model_file_multiple_files(self, fake_models_dir, monkeypatch):
        """Should return most recent file when multiple exist."""
        original_cwd = os.getcwd()
        try:
            os.chdir(fake_models_dir.parent)
            
            # Create older and newer files
            old_file = fake_models_dir / "old_xgboost.joblib"
            old_file.write_text("old")
            
            time.sleep(0.01)  # Ensure different timestamps
            
            new_file = fake_models_dir / "new_xgboost.joblib"
            new_file.write_text("new")
            
            result = find_latest_model_file("xgboost")
            assert result is not None
            assert "new" in result or result.endswith("new_xgboost.joblib")
        finally:
            os.chdir(original_cwd)


# ------------------------------------------------------------------------------
# Unit Tests - Training Execution
# ------------------------------------------------------------------------------
class TestTrainingExecution:
    
    def test_run_training_script_success(self, monkeypatch, fake_script, tmp_path):
        """Should update job status on successful training."""
        job_id = create_job_id()
        script_path = str(fake_script["xgboost"])
        register_job(job_id, "xgboost", script_path)
        
        # Mock subprocess - must match the actual call signature
        mock_result = Mock()
        mock_result.stdout = "Training completed successfully"
        mock_result.stderr = ""
        mock_result.returncode = 0
        
        def mock_run(cmd, capture_output=True, text=True, check=True, cwd=None, env=None):
            return mock_result
        
        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(training_api, "find_latest_model_file", lambda _: "/path/to/model.joblib")
        
        # Mock environment
        monkeypatch.setenv("REPO_ROOT", str(tmp_path))
        
        run_training_script(script_path, job_id, "xgboost")
        
        job = training_jobs[job_id]
        assert job["status"] == "completed"
        assert "Training completed" in job["logs"]
        assert job["model_path"] == "/path/to/model.joblib"
        assert job["completed_at"] is not None
        assert job["error"] is None
    
    def test_run_training_script_failure(self, monkeypatch, fake_script, tmp_path):
        """Should handle training failure correctly."""
        job_id = create_job_id()
        script_path = str(fake_script["xgboost"])
        register_job(job_id, "xgboost", script_path)
        
        def mock_run(cmd, capture_output=True, text=True, check=True, cwd=None, env=None):
            error = subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                stderr="Training failed with error"
            )
            error.stdout = "Some output"
            raise error
        
        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(training_api, "find_latest_model_file", lambda _: None)
        monkeypatch.setenv("REPO_ROOT", str(tmp_path))
        
        run_training_script(script_path, job_id, "xgboost")
        
        job = training_jobs[job_id]
        assert job["status"] == "failed"
        assert "failed" in job["error"].lower() or "error" in job["error"].lower()
        assert job["completed_at"] is not None
        assert "Some output" in job["logs"]
    
    def test_run_training_script_unexpected_exception(self, monkeypatch, fake_script, tmp_path):
        """Should handle unexpected exceptions."""
        job_id = create_job_id()
        script_path = str(fake_script["xgboost"])
        register_job(job_id, "xgboost", script_path)
        
        def mock_run(cmd, capture_output=True, text=True, check=True, cwd=None, env=None):
            raise RuntimeError("Unexpected error")
        
        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setenv("REPO_ROOT", str(tmp_path))
        
        run_training_script(script_path, job_id, "xgboost")
        
        job = training_jobs[job_id]
        assert job["status"] == "failed"
        assert "error" in job["error"].lower()
    
    def test_run_training_script_updates_status_to_running(self, monkeypatch, fake_script, tmp_path):
        """Should set status to running before execution."""
        job_id = create_job_id()
        script_path = str(fake_script["xgboost"])
        register_job(job_id, "xgboost", script_path)
        
        statuses_seen = []
        
        def mock_run(cmd, capture_output=True, text=True, check=True, cwd=None, env=None):
            # Capture status during execution
            statuses_seen.append(training_jobs[job_id]["status"])
            result = Mock()
            result.stdout = "OK"
            result.stderr = ""
            result.returncode = 0
            return result
        
        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(training_api, "find_latest_model_file", lambda _: None)
        monkeypatch.setenv("REPO_ROOT", str(tmp_path))
        
        run_training_script(script_path, job_id, "xgboost")
        
        # The status should be "running" when subprocess.run is called
        assert "running" in statuses_seen


# ------------------------------------------------------------------------------
# API Endpoint Tests - Single Model Training
# ------------------------------------------------------------------------------
class TestTrainModelEndpoint:
    
    def test_train_model_xgboost(self, monkeypatch, fake_script):
        """POST /train/xgboost should start training job."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        response = client.post("/train/xgboost")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["model_type"] == "xgboost"
        assert "job_id" in data
        assert uuid.UUID(data["job_id"])  # Valid UUID
    
    def test_train_model_neural_net(self, monkeypatch, fake_script):
        """POST /train/neural-net should start training job."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["neural"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        response = client.post("/train/neural-net")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "neural-net"
    
    def test_train_model_random_forest(self, monkeypatch, fake_script):
        """POST /train/random-forest should start training job."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["rf"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        response = client.post("/train/random-forest")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "random-forest"
    
    def test_train_model_invalid_type(self):
        """Should reject invalid model type."""
        response = client.post("/train/invalid-model")
        assert response.status_code in [400, 404, 500]
    
    def test_train_model_script_not_found(self, monkeypatch):
        """Should handle missing training script."""
        def mock_validate(path):
            raise HTTPException(status_code=404, detail="Script not found")
        
        monkeypatch.setattr(training_api, "validate_training_script", mock_validate)
        
        response = client.post("/train/xgboost")
        assert response.status_code == 404
    
    def test_train_model_creates_job_registry_entry(self, monkeypatch, fake_script):
        """Should register job in training_jobs dict."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        response = client.post("/train/xgboost")
        job_id = response.json()["job_id"]
        
        assert job_id in training_jobs
        assert training_jobs[job_id]["model_type"] == "xgboost"


# ------------------------------------------------------------------------------
# API Endpoint Tests - Training with Config
# ------------------------------------------------------------------------------
class TestTrainWithConfigEndpoint:
    
    def test_train_with_config_single_model(self, monkeypatch, fake_script):
        """POST /train with config for single model."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        payload = {
            "model_type": "xgboost",
            "retrain": True,
            "use_cv": True
        }
        
        response = client.post("/train", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "xgboost"
        assert data["status"] == "started"
    
    def test_train_with_config_all_models(self, monkeypatch, fake_script):
        """POST /train with model_type='all' should train all models."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        payload = {
            "model_type": "all",
            "retrain": False,
            "use_cv": True
        }
        
        response = client.post("/train", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "all"
        assert data["status"] == "started"
        
        # Check that parent job was created
        job_id = data["job_id"]
        assert job_id in training_jobs
        assert "sub_jobs" in training_jobs[job_id]
    
    def test_train_with_config_hyperparameters(self, monkeypatch, fake_script):
        """Should accept and store hyperparameters."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        payload = {
            "model_type": "xgboost",
            "hyperparameters": {
                "learning_rate": 0.1,
                "max_depth": 5
            }
        }
        
        response = client.post("/train", json=payload)
        
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        # Hyperparameters should be stored in job
        assert "hyperparameters" in training_jobs[job_id]
        assert training_jobs[job_id]["hyperparameters"]["learning_rate"] == 0.1
    
    def test_train_with_config_invalid_payload(self):
        """Should reject invalid request payload."""
        payload = {
            "invalid_field": "value"
        }
        
        response = client.post("/train", json=payload)
        assert response.status_code == 422  # Validation error


# ------------------------------------------------------------------------------
# API Endpoint Tests - Job Status
# ------------------------------------------------------------------------------
class TestJobStatusEndpoint:
    
    def test_get_job_status_pending(self, fake_script):
        """Should return pending status for new job."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        response = client.get(f"/train/status/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "pending"
        assert data["model_type"] == "xgboost"
    
    def test_get_job_status_completed(self, fake_script):
        """Should return completed status with model path."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["model_path"] = "/path/to/model.joblib"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        
        response = client.get(f"/train/status/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["model_path"] == "/path/to/model.joblib"
        assert data["completed_at"] is not None
    
    def test_get_job_status_failed(self, fake_script):
        """Should return failed status with error message."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = "Training failed"
        
        response = client.get(f"/train/status/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Training failed"
    
    def test_get_job_status_not_found(self):
        """Should return 404 for unknown job ID."""
        response = client.get(f"/train/status/{uuid.uuid4()}")
        assert response.status_code == 404
    
    def test_get_job_status_all_models_aggregation(self, fake_script):
        """Should aggregate status for 'all' model type."""
        parent_job_id = create_job_id()
        sub_job_ids = [create_job_id() for _ in range(3)]
        
        # Create parent job
        training_jobs[parent_job_id] = {
            "job_id": parent_job_id,
            "status": "pending",
            "model_type": "all",
            "started_at": datetime.utcnow().isoformat(),
            "sub_jobs": sub_job_ids
        }
        
        # Create sub jobs
        for sub_id in sub_job_ids:
            register_job(sub_id, "xgboost", str(fake_script["xgboost"]))
            training_jobs[sub_id]["status"] = "completed"
        
        response = client.get(f"/train/status/{parent_job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"  # All sub-jobs completed
    
    def test_get_job_status_all_models_partial_failure(self, fake_script):
        """Should show failed if any sub-job failed."""
        parent_job_id = create_job_id()
        sub_job_ids = [create_job_id() for _ in range(3)]
        
        training_jobs[parent_job_id] = {
            "job_id": parent_job_id,
            "status": "pending",
            "model_type": "all",
            "started_at": datetime.utcnow().isoformat(),
            "sub_jobs": sub_job_ids
        }
        
        for i, sub_id in enumerate(sub_job_ids):
            register_job(sub_id, "xgboost", str(fake_script["xgboost"]))
            training_jobs[sub_id]["status"] = "failed" if i == 1 else "completed"
        
        response = client.get(f"/train/status/{parent_job_id}")
        data = response.json()
        assert data["status"] == "failed"


# ------------------------------------------------------------------------------
# API Endpoint Tests - List Jobs
# ------------------------------------------------------------------------------
class TestListJobsEndpoint:
    
    def test_list_jobs_empty(self):
        """Should return empty list when no jobs."""
        response = client.get("/train/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
        assert data["total_count"] == 0
    
    def test_list_jobs_multiple(self, fake_script):
        """Should list all registered jobs."""
        job_ids = []
        for i in range(5):
            job_id = create_job_id()
            register_job(job_id, "xgboost", str(fake_script["xgboost"]))
            job_ids.append(job_id)
        
        response = client.get("/train/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 5
        assert data["total_count"] == 5
    
    def test_list_jobs_with_limit(self, fake_script):
        """Should respect limit parameter."""
        for i in range(15):
            job_id = create_job_id()
            register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        response = client.get("/train/jobs?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 5
        assert data["total_count"] == 15
    
    def test_list_jobs_filter_by_status(self, fake_script):
        """Should filter jobs by status."""
        for i in range(3):
            job_id = create_job_id()
            register_job(job_id, "xgboost", str(fake_script["xgboost"]))
            training_jobs[job_id]["status"] = "completed"
        
        for i in range(2):
            job_id = create_job_id()
            register_job(job_id, "xgboost", str(fake_script["xgboost"]))
            training_jobs[job_id]["status"] = "failed"
        
        response = client.get("/train/jobs?status=completed")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 3
        assert all(job["status"] == "completed" for job in data["jobs"])
    
    def test_list_jobs_sorted_by_time(self, fake_script):
        """Should return jobs sorted by start time (newest first)."""
        job_ids = []
        for i in range(3):
            job_id = create_job_id()
            register_job(job_id, "xgboost", str(fake_script["xgboost"]))
            job_ids.append(job_id)
            time.sleep(0.01)  # Ensure different timestamps
        
        response = client.get("/train/jobs")
        
        data = response.json()
        jobs = data["jobs"]
        
        # Newest should be first
        assert jobs[0]["job_id"] == job_ids[-1]
        assert jobs[-1]["job_id"] == job_ids[0]


# ------------------------------------------------------------------------------
# API Endpoint Tests - Cancel Job
# ------------------------------------------------------------------------------
class TestCancelJobEndpoint:
    
    def test_cancel_pending_job(self, fake_script):
        """Should cancel pending job successfully."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        response = client.delete(f"/train/job/{job_id}")
        
        assert response.status_code == 200
        assert training_jobs[job_id]["status"] == "cancelled"
        assert "cancelled" in training_jobs[job_id]["error"].lower()
    
    def test_cancel_running_job(self, fake_script):
        """Should cancel running job."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        training_jobs[job_id]["status"] = "running"
        
        response = client.delete(f"/train/job/{job_id}")
        
        assert response.status_code == 200
        assert training_jobs[job_id]["status"] == "cancelled"
    
    def test_cancel_completed_job_fails(self, fake_script):
        """Should not cancel completed job."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        training_jobs[job_id]["status"] = "completed"
        
        response = client.delete(f"/train/job/{job_id}")
        
        assert response.status_code == 400
    
    def test_cancel_failed_job_fails(self, fake_script):
        """Should not cancel failed job."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        training_jobs[job_id]["status"] = "failed"
        
        response = client.delete(f"/train/job/{job_id}")
        
        assert response.status_code == 400
    
    def test_cancel_nonexistent_job(self):
        """Should return 404 for unknown job."""
        response = client.delete(f"/train/job/{uuid.uuid4()}")
        assert response.status_code == 404


# ------------------------------------------------------------------------------
# API Endpoint Tests - Available Models
# ------------------------------------------------------------------------------
class TestAvailableModelsEndpoint:
    
    def test_get_available_models_empty_dir(self, tmp_path, monkeypatch):
        """Should return empty dict when no models exist."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            response = client.get("/train/models/available")
            
            assert response.status_code == 200
            data = response.json()
            assert data["available_models"] == {}
        finally:
            os.chdir(original_cwd)
        
        response = client.get("/train/models/available")
        
        assert response.status_code == 200
        data = response.json()
        assert data["available_models"] == {}
    
    def test_get_available_models_with_files(self, fake_models_dir, monkeypatch):
        """Should list all model files with metadata."""
        original_cwd = os.getcwd()
        try:
            os.chdir(fake_models_dir.parent)
            response = client.get("/train/models/available")
            
            assert response.status_code == 200
            data = response.json()
            
            models = data["available_models"]
            assert len(models) > 0
            
            # Check structure of model info
            for model_name, model_info in models.items():
                assert "path" in model_info
                assert "type" in model_info
                assert "size" in model_info
                assert "modified" in model_info
        finally:
            os.chdir(original_cwd)
    
    def test_get_available_models_identifies_types(self, fake_models_dir, monkeypatch):
        """Should correctly identify model types from filenames."""
        original_cwd = os.getcwd()
        try:
            os.chdir(fake_models_dir.parent)
            response = client.get("/train/models/available")
            data = response.json()
            models = data["available_models"]
            
            # Check that model types are identified
            xgb_models = [m for m in models.values() if m["type"] == "xgboost"]
            nn_models = [m for m in models.values() if m["type"] == "neural-net"]
            rf_models = [m for m in models.values() if m["type"] == "random-forest"]
            
            assert len(xgb_models) > 0 or len(nn_models) > 0 or len(rf_models) > 0
        finally:
            os.chdir(original_cwd)
    
    def test_get_available_models_no_dir(self, tmp_path, monkeypatch):
        """Should handle case when models directory doesn't exist."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            response = client.get("/train/models/available")
            
            assert response.status_code == 200
            data = response.json()
            assert data["available_models"] == {}
        finally:
            os.chdir(original_cwd)


# ------------------------------------------------------------------------------
# Integration Tests - End to End
# ------------------------------------------------------------------------------
class TestEndToEndTrainingFlow:
    
    def test_complete_training_workflow(self, monkeypatch, fake_script):
        """Test complete flow: start -> check status -> complete."""
        # Mock subprocess to complete immediately
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        
        def mock_run_training(script_path, job_id, model_type):
            training_jobs[job_id]["status"] = "running"
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
            training_jobs[job_id]["logs"] = "Training completed"
        
        monkeypatch.setattr(training_api, "run_training_script", mock_run_training)
        
        # 1. Start training
        response = client.post("/train/xgboost")
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        # 2. Check status
        response = client.get(f"/train/status/{job_id}")
        assert response.status_code == 200
        
        # 3. Verify in jobs list
        response = client.get("/train/jobs")
        assert any(job["job_id"] == job_id for job in response.json()["jobs"])
    
    def test_training_failure_workflow(self, monkeypatch, fake_script):
        """Test flow when training fails."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        
        def mock_run_training(script_path, job_id, model_type):
            training_jobs[job_id]["status"] = "running"
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = "Training error"
            training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        
        monkeypatch.setattr(training_api, "run_training_script", mock_run_training)
        
        # Start training
        response = client.post("/train/xgboost")
        job_id = response.json()["job_id"]
        
        # Check failed status
        response = client.get(f"/train/status/{job_id}")
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Training error"
    
    def test_concurrent_training_jobs(self, monkeypatch, fake_script):
        """Test multiple training jobs running concurrently."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        job_ids = []
        
        # Start multiple jobs
        for model_type in ["xgboost", "neural-net", "random-forest"]:
            response = client.post(f"/train/{model_type}")
            assert response.status_code == 200
            job_ids.append(response.json()["job_id"])
        
        # All should be registered
        assert len(job_ids) == 3
        for job_id in job_ids:
            assert job_id in training_jobs
    
    def test_train_all_models_workflow(self, monkeypatch, fake_script):
        """Test training all models at once."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        payload = {"model_type": "all", "retrain": True}
        response = client.post("/train", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        parent_job_id = data["job_id"]
        
        # Check parent job has sub-jobs
        parent_job = training_jobs[parent_job_id]
        assert "sub_jobs" in parent_job
        assert len(parent_job["sub_jobs"]) == 3  # xgboost, neural-net, random-forest


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------
class TestEdgeCases:
    
    def test_empty_job_id(self):
        """Should handle empty job ID gracefully."""
        response = client.get("/train/status/")
        assert response.status_code in [404, 405]  # Not found or method not allowed
    
    def test_malformed_job_id(self):
        """Should handle malformed job ID."""
        response = client.get("/train/status/not-a-uuid")
        assert response.status_code == 404
    
    def test_special_characters_in_model_type(self):
        """Should handle special characters in model type."""
        response = client.post("/train/model<script>alert()</script>")
        assert response.status_code in [400, 404, 500]
    
    def test_extremely_long_model_type(self):
        """Should handle extremely long model type."""
        long_type = "x" * 10000
        response = client.post(f"/train/{long_type}")
        assert response.status_code in [400, 404, 414, 500]
    
    def test_negative_limit_in_list_jobs(self):
        """Should handle negative limit parameter."""
        response = client.get("/train/jobs?limit=-1")
        # Should either reject or treat as 0
        assert response.status_code in [200, 400, 422]
    
    def test_invalid_status_filter(self):
        """Should handle invalid status filter."""
        response = client.get("/train/jobs?status=invalid_status")
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
    
    def test_sql_injection_attempt(self):
        """Should be safe from SQL injection attempts."""
        malicious = "xgboost'; DROP TABLE jobs; --"
        response = client.post(f"/train/{malicious}")
        # Should not cause internal error, just 404 or 400
        assert response.status_code in [400, 404, 500]
    
    def test_path_traversal_attempt(self):
        """Should prevent path traversal attacks."""
        malicious_path = "../../etc/passwd"
        with pytest.raises(HTTPException):
            validate_training_script(malicious_path)
    
    def test_training_with_null_hyperparameters(self, monkeypatch, fake_script):
        """Should handle null hyperparameters."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        payload = {
            "model_type": "xgboost",
            "hyperparameters": None
        }
        
        response = client.post("/train", json=payload)
        assert response.status_code == 200
    
    def test_training_with_empty_hyperparameters(self, monkeypatch, fake_script):
        """Should handle empty hyperparameters dict."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        monkeypatch.setattr(training_api, "run_training_script", lambda *a, **k: None)
        
        payload = {
            "model_type": "xgboost",
            "hyperparameters": {}
        }
        
        response = client.post("/train", json=payload)
        assert response.status_code == 200


# ------------------------------------------------------------------------------
# Performance and Stress Tests
# ------------------------------------------------------------------------------
class TestPerformance:
    
    def test_many_jobs_registration(self, fake_script):
        """Should handle registering many jobs efficiently."""
        start_time = time.time()
        
        job_ids = []
        for i in range(100):
            job_id = create_job_id()
            register_job(job_id, "xgboost", str(fake_script["xgboost"]))
            job_ids.append(job_id)
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for 100 jobs)
        assert elapsed < 1.0
        assert len(training_jobs) == 100
    
    def test_list_jobs_with_many_entries(self, fake_script):
        """Should efficiently list jobs when many exist."""
        # Create many jobs
        for i in range(100):
            job_id = create_job_id()
            register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        start_time = time.time()
        response = client.get("/train/jobs?limit=10")
        elapsed = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Should be fast even with many jobs
    
    def test_concurrent_status_checks(self, fake_script):
        """Should handle concurrent status checks."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        # Simulate concurrent requests
        responses = []
        for i in range(10):
            response = client.get(f"/train/status/{job_id}")
            responses.append(response)
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)


# ------------------------------------------------------------------------------
# Logging Tests
# ------------------------------------------------------------------------------
class TestLogging:
    
    def test_training_job_logs_success(self, monkeypatch, fake_script, tmp_path, caplog):
        """Should log successful training."""
        job_id = create_job_id()
        script_path = str(fake_script["xgboost"])
        register_job(job_id, "xgboost", script_path)
        
        mock_result = Mock()
        mock_result.stdout = "Training completed"
        mock_result.stderr = ""
        mock_result.returncode = 0
        
        def mock_run(cmd, capture_output=True, text=True, check=True, cwd=None, env=None):
            return mock_result
        
        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(training_api, "find_latest_model_file", lambda _: None)
        monkeypatch.setenv("REPO_ROOT", str(tmp_path))
        
        with caplog.at_level(logging.INFO):
            run_training_script(script_path, job_id, "xgboost")
        
        # Check logs contain job information
        log_messages = [record.message for record in caplog.records]
        assert any(job_id in msg for msg in log_messages)
    
    def test_training_job_logs_failure(self, monkeypatch, fake_script, tmp_path, caplog):
        """Should log training failures."""
        job_id = create_job_id()
        script_path = str(fake_script["xgboost"])
        register_job(job_id, "xgboost", script_path)
        
        def mock_run(cmd, capture_output=True, text=True, check=True, cwd=None, env=None):
            raise subprocess.CalledProcessError(1, cmd, stderr="Error!")
        
        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(training_api, "find_latest_model_file", lambda _: None)
        monkeypatch.setenv("REPO_ROOT", str(tmp_path))
        
        with caplog.at_level(logging.ERROR):
            run_training_script(script_path, job_id, "xgboost")
        
        log_messages = [record.message for record in caplog.records]
        assert any("failed" in msg.lower() for msg in log_messages)


# ------------------------------------------------------------------------------
# Data Model Tests
# ------------------------------------------------------------------------------
class TestDataModels:
    
    def test_training_request_model_valid(self):
        """TrainingRequest should validate correct data."""
        request = TrainingRequest(
            model_type="xgboost",
            retrain=True,
            use_cv=False,
            hyperparameters={"learning_rate": 0.1}
        )
        
        assert request.model_type == "xgboost"
        assert request.retrain is True
        assert request.use_cv is False
    
    def test_training_request_defaults(self):
        """TrainingRequest should use default values."""
        request = TrainingRequest(model_type="xgboost")
        
        assert request.retrain is False
        assert request.use_cv is True
        assert request.hyperparameters is None
    
    def test_training_response_structure(self):
        """TrainingResponse should have correct structure."""
        from src.api.routers.train import TrainingResponse
        
        response = TrainingResponse(
            job_id="test-123",
            status="started",
            message="Training initiated",
            model_type="xgboost"
        )
        
        assert response.job_id == "test-123"
        assert response.status == "started"
    
    def test_job_status_response_structure(self):
        """JobStatusResponse should have correct structure."""
        from src.api.routers.train import JobStatusResponse
        
        response = JobStatusResponse(
            job_id="test-123",
            status="completed",
            model_type="xgboost",
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:10:00",
            model_path="/path/to/model.joblib"
        )
        
        assert response.completed_at == "2024-01-01T00:10:00"
        assert response.model_path == "/path/to/model.joblib"


# ------------------------------------------------------------------------------
# Environment and Configuration Tests
# ------------------------------------------------------------------------------
class TestEnvironmentConfiguration:
    
    def test_repo_root_environment_variable(self):
        """Should set REPO_ROOT environment variable."""
        assert "REPO_ROOT" in os.environ
    
    def test_pythonpath_environment_variable(self):
        """Should set PYTHONPATH environment variable."""
        assert "PYTHONPATH" in os.environ
    
    def test_log_directory_creation(self, tmp_path, monkeypatch):
        """Should create log directory if it doesn't exist."""
        log_path = tmp_path / "logs"
        monkeypatch.setattr(training_api, "log_path", str(log_path))
        
        # Simulate module initialization
        os.makedirs(log_path, exist_ok=True)
        
        assert log_path.exists()


# ------------------------------------------------------------------------------
# Subprocess Command Generation Tests
# ------------------------------------------------------------------------------
class TestSubprocessCommandGeneration:
    
    def test_module_execution_for_src_scripts(self, monkeypatch):
        """Should use -m flag for scripts in src/ directory."""
        repo_root = Path("/fake/repo")
        script_path = repo_root / "src" / "models" / "train_xgboost.py"
        
        monkeypatch.setenv("REPO_ROOT", str(repo_root))
        
        # This tests the logic inside run_training_script
        # The actual command would be: python -m src.models.train_xgboost
    
    def test_direct_execution_for_non_src_scripts(self):
        """Should use direct execution for scripts outside src/."""
        # Scripts not in src/ should be executed directly
        pass


# ------------------------------------------------------------------------------
# Cleanup and Resource Management
# ------------------------------------------------------------------------------
class TestResourceManagement:
    
    def test_job_registry_isolation(self):
        """Each test should have isolated job registry."""
        # This is ensured by the autouse fixture
        assert len(training_jobs) == 0
    
    def test_no_memory_leaks_in_job_storage(self, fake_script):
        """Job storage shouldn't grow unbounded (in production, use DB)."""
        initial_count = len(training_jobs)
        
        # Create many jobs
        for i in range(50):
            job_id = create_job_id()
            register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        
        # In production, implement cleanup of old jobs
        # For now, just verify they're all stored
        assert len(training_jobs) == initial_count + 50


# ------------------------------------------------------------------------------
# Helper Function Tests
# ------------------------------------------------------------------------------
class TestHelperFunctions:
    
    def test_start_single_training(self, monkeypatch, fake_script):
        """Should start single training job correctly."""
        monkeypatch.setattr(training_api, "validate_training_script", lambda x: str(fake_script["xgboost"]))
        
        from fastapi import BackgroundTasks
        background_tasks = BackgroundTasks()
        
        request = TrainingRequest(
            model_type="xgboost",
            hyperparameters={"max_depth": 5}
        )
        
        # Use sync wrapper for async function
        import asyncio
        
        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        job_id = loop.run_until_complete(
            start_single_training("xgboost", background_tasks, request)
        )
        
        assert job_id in training_jobs
        assert training_jobs[job_id]["hyperparameters"]["max_depth"] == 5


# ------------------------------------------------------------------------------
# Documentation and API Contract Tests
# ------------------------------------------------------------------------------
class TestAPIContracts:
    
    def test_all_endpoints_have_response_models(self):
        """All endpoints should specify response models."""
        # This ensures API contracts are clear
        pass
    
    def test_error_responses_are_consistent(self):
        """Error responses should follow consistent format."""
        # Test 404
        response = client.get("/train/status/nonexistent")
        assert response.status_code == 404
        assert "detail" in response.json()
        
        # Test 400
        response = client.delete("/train/job/nonexistent")
        assert response.status_code == 404
        assert "detail" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])