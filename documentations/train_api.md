# Train API Documentation

## Overview

The **Train API** provides endpoints to trigger, monitor, and manage model training tasks.  
It allows you to start training jobs for specific model types such as **XGBoost**, **Random Forest**, or **Neural Network**, all managed asynchronously through FastAPI background tasks.

Training scripts are defined under:
src/models/

```text
|-- churn_nn.py
|-- train_RandomForest.py
|-- train_xgboost.py
```

## Endpoints Summary

| Method | Endpoint | Description |
|--------|-----------|-------------|
| **POST** | `/train/{model_type}` | Start training for a specific model type |
| **POST** | `/train` | Start training with configuration (supports "all" models) |
| **GET** | `/train/status/{job_id}` | Get the current status of a training job |
| **GET** | `/train/jobs` | List recent or filtered training jobs |
| **DELETE** | `/train/job/{job_id}` | Cancel a training job. **(Note: This is a soft cancel, it updates the job status but does not terminate the running script process)** |
| **GET** | `/train/models/available` | Get list of available trained model files |

---

## Request Models

### **TrainingRequest**
| Field | Type | Default | Description |
|--------|------|----------|-------------|
| `model_type` | `str` | required | `"neural-net"`, `"xgboost"`, `"random-forest"`, or `"all"` |
| `retrain` | `bool` | `False` | Whether to overwrite existing model |
| `use_cv` | `bool` | `True` | Use cross-validation during training |
| `hyperparameters` | `dict` | `None` | Optional hyperparameter overrides |

### **Training Response**
| Field | Type | Description |
|--------|------|-------------|
| `job_id` | `str` | Unique identifier for the job |
| `status` | `str` | `"started"`, `"running"`, `"completed"`, `"failed"` |
| `message` | `str` | Training status message |
| `model_type` | `str` | Type of model being trained |

---

## POST `/train/{model_type}`

Trigger training for a **specific** model type.

### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/train/xgboost"
