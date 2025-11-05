# Predict API Documentation

## Overview

The **Predict API** provides endpoints to generate predictions using trained models.  
It loads the most recent model of the specified type (e.g., XGBoost, Random Forest, Neural Network) and performs inference using input features provided in the `predict_input.json` file.

- **Prediction logic location:** `src/api/routers/predict.py`  
- **Models directory:** `models/`  

---

## Endpoints Summary

| Method | Endpoint | Description |
|--------|---------|-------------|
| **POST** | `/predict/{model_type}` | Generate predictions using the specified model type. |

---

## Supported Models

| Model Type | File Extension | Example Filename |
|------------|----------------|----------------|
| `xgboost` | `.joblib`       | `xgboost_model_2025-10-30.joblib` |
| `random-forest` | `.joblib` | `rf_model_2025-10-28.joblib` |
| `neural-net` | `.pth`       | `nn_model_2025-10-29.pth` |

**Note:** The API automatically selects the latest model for the given type based on the filename timestamp.

---

## Request Format

Before making a prediction request, ensure the following JSON file exists:

`src/api/routers/predict_input.json`

## Example content:

```json
{
  "features": {
    "revenue": 102.5,
    "mou": 189.0,
    "overage": 5.2,
    "roam": 1.0
    // include all other required features
  }
}
```
## Example Request
### Using curl to call the Predict API:
```bash
curl -X POST "http://127.0.0.1:8000/train/xgboost"
```

## Expected Response
### Using curl to call the Predict API:
```json
{
  "model_type": "xgboost",
  "prediction": [1],
  "model_path": "models/xgboost_model_2025-10-30.joblib"
}
```
## Error Responses
| Status Code | Description                                                 | Example                                                                                     |
| ----------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `404`       | No trained model found or missing `predict_input.json` file | `"detail": "predict_input.json not found"`                                                  |
| `400`       | Invalid model type                                          | `"detail": "Invalid model_type. Must be one of ['xgboost', 'neural-net', 'random-forest']"` |
