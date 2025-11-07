# API Overview

---

## Introduction

This API provides access to the **Machine Learning Model Management System**, built using **FastAPI**.  
It enables training, prediction, evaluation, and tracking of multiple ML models including:
- **XGBoost**
- **Random Forest**
- **Neural Network**

The system allows:
- Triggering and monitoring **training jobs**
- Making **real-time predictions**
- Accessing **model metrics and performance reports**
- Managing **versioned model files** in the `/models` directory

The API is designed for **scalability, modularity, and reproducibility**, allowing multiple models to coexist with independent versions.

---

## API Architecture

The system follows a modular structure:


Each API module defines endpoints under its respective router (e.g., `/train`, `/predict`, `/metrics`, `/validation`).

---

## Base URL

All endpoints are accessible via: http://127.0.0.1:8000/


---

## Core Endpoints

| Category | Endpoint | Method | Description |
|-----------|-----------|--------|-------------|
| **Train API** | `/train/{model_type}` | `POST` | Trigger training for a specific model |
|  | `/train` | `POST` | Train multiple models or using a config file |
|  | `/train/status/{job_id}` | `GET` | Retrieve job status |
| **Predict API** | `/predict/{model_type}` | `POST` | Make predictions using a trained model |
|  | `/predict/latest/{model_type}` | `GET` | Get latest model info |
| **Metrics API** | `/metrics/{model_type}` | `GET` | Retrieve model metrics |
|  | `/metrics/all` | `GET` | Compare metrics for all models |
| **Data Validation API** | `/data_validation/validate` | `POST` | Validate dataset schema against model schema |

---

## Supported Model Types

| Model Type | File Extension | Framework | Location |
|-------------|----------------|-----------|-----------|
| `xgboost` | `.joblib` | XGBoost | `models/xgboost_final_model.joblib` |
| `random-forest` | `.joblib` | Scikit-learn | `models/random_forest_final_model.joblib` |
| `neural-net` | `.pth` | PyTorch | `models/neural_net_final_model.pth` |

---

## Common Response Format

All endpoints return a unified JSON structure:

```json
{
  "status": "success",
  "message": "Action completed successfully.",
  "data": {
    "model_type": "xgboost",
    "timestamp": "2025-10-30T10:00:00Z"
  }
}
```


### Error Responses
```json
{
  "status": "error",
  "message": "Model file not found.",
  "detail": "xgboost_final_model.joblib is missing from models directory"
}
```
### Testing the API
### You can test the endpoints using:
```text
Swagger UI → http://127.0.0.1:8000/docs
ReDoc → http://127.0.0.1:8000/redoc
cURL / HTTPie / Postman
```

 ### Example (using cURL):
```bash
curl -X GET "http://127.0.0.1:8000/metrics/all"
```
| Document                           | Description                    |
| ---------------------------------- | ------------------------------ |
| [train_api.md](./train_API.md)     | Model training endpoints       |
| [predict_api.md](./predict_API.md) | Model inference endpoints      |
| [metrics_api.md](./metrics_API.md) | Metrics and evaluation         |
| [examples.md](./examples.md)       | Example requests and responses |
| [validation.md](./validation.md)   | Data validation endpoints      |

