# Model Metrics API Documentation

## Overview

The **Model Metrics API** provides an endpoint to evaluate trained machine learning models using stored test data.  
It loads the most recent version of the specified model type (e.g., XGBoost, Random Forest, Neural Network) and computes performance metrics such as **Accuracy**, **F1 Score**, and **ROC AUC**.

- **Metrics logic location:** `src/api/routers/metrics.py`  
- **Models directory:** `models/`  
- **Test data file:** `test_input.json`
**Note:** The test data must follow the same schema as the `CustomerData` payload used in `/predict`.
Each record should have a `features` dictionary matching model inputs and a `target` field for the true label.
---

## Endpoints Summary

| Method | Endpoint | Description |
|--------|-----------|-------------|
| **POST** | `/metrics/{model_type}` | Evaluate the specified model on test data and return performance metrics. |

---

## Supported Models

| Model Type | File Extension | Example Filename |
|-------------|----------------|------------------|
| `xgboost` | `.joblib` | `xgboost_model_2025-10-30.joblib` |
| `random-forest` | `.joblib` | `rf_model_2025-10-28.joblib` |
| `neural-net` | `.pth` | `nn_model_2025-10-29.pth` |

**Note:** The API automatically selects the most recently saved model file for the chosen type.

---
## Request Format
## Test Data Format
### Before making a metrics request, ensure the following JSON file exists:
`test_input.json`

- Each record must follow the `CustomerData` schema.
- The `features` dictionary contains all input features the model expects.
- The `target` field is the true label for evaluation.

### Example content:
```json
[
  {
    "features": {
      "revenue": 102.5,
      "mou": 189.0,
      "overage": 5.2,
      "roam": 1.0
      // include all other required features
    },
    "target": 1
  },
  {
    "features": {
      "revenue": 84.3,
      "mou": 95.0,
      "overage": 1.5,
      "roam": 0.0
    },
    "target": 0
  }
]
```
**Note**: Simulated test data can be generated using the `data_simulation.py` or similar scripts. Just point `config.test_data_path` to that file.

## Expected Response
```json
{
  "model_type": "xgboost",
  "model_path": "models/xgboost_model_2025-10-30.joblib",
  "metrics": {
    "accuracy": 0.84,
    "f1_score": 0.79,
    "roc_auc": 0.85
  },
  "test_samples": 100
}
```
## Error Responses
| Status Code | Description                       | Example                                                                                     |
| ----------- | --------------------------------- | ------------------------------------------------------------------------------------------- |
| `404`       | Test data file or model not found | `"detail": "test_input.json not found"`                                                     |
| `400`       | Invalid model type                | `"detail": "Invalid model_type. Must be one of ['xgboost', 'neural-net', 'random-forest']"` |

## Data Flow Diagram
         ┌────────────────────┐
         │  /predict Payload  │
         │   CustomerData     │
         │  (Single Record)   │
         └─────────┬─────────┘
                   │
                   │ Schema
                   │
                   ▼
         ┌────────────────────┐
         │ Model Input Features│
         │   (X) for model)   │
         └─────────┬─────────┘
                   │
           ┌───────┴────────┐
           │   Trained Model │
           │ (XGBoost, RF,  │
           │ Neural Net)     │
           └───────┬────────┘
                   │
                   │ Predictions
                   ▼
         ┌────────────────────┐
         │   /metrics Test    │
         │    Data JSON       │
         │ (Multiple Records) │
         │ features + target  │
         └─────────┬─────────┘
                   │
                   │ Evaluation
                   ▼
         ┌────────────────────┐
         │   Performance       │
         │   Metrics           │
         │ Accuracy, F1, ROC  │
         └────────────────────┘
