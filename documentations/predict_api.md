# Predict API Documentation

## Overview

The **Predict API** provides endpoints to generate predictions for customer churn using trained models.  
It loads the most recent model of the specified type (e.g., XGBoost, Random Forest, Neural Network) and performs inference using input features provided via JSON payload or from the database.

- **Prediction logic location:** `src/api/routers/predict.py`  
- **Models directory:** `models/`  

---

## Endpoints Summary
| Method   | Endpoint                                       | Description                                                      |
| -------- | ---------------------------------------------- | ---------------------------------------------------------------- |
| **POST** | `/predict/{model_type}`                        | Generate predictions from a JSON payload.                        |
| **GET**  | `/predict/{model_type}/customer/{customer_id}` | Generate predictions for a specific customer from the database.  |
| **GET**  | `/predict/{model_type}/batch/{batch_id}`       | Generate predictions for a batch of customers from the database. |

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



## Example JSON payload::
1. **POST** `/predict/{model_type}` **(From JSON Payload)**
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
**Notes:**

- If the `churn` field is provided, it will be returned as the prediction.
- Non-negative numeric fields: revenue, mou, months, actvsubs, uniqsubs, etc.
- Binary categorical fields must be`"0"` or `"1"`: e.g., `newcelly`, `newcelln`, `children`, `creditcd`, etc.
- `newcelly` and `newcelln` are mutually exclusive; exactly one must be set.
- `age1` and `age2` must satisfy `0 <= age <= 120` and `age1 >= age2.`

## Example curl Request
```bash
curl -X POST "http://127.0.0.1:8000/predict/xgboost" \
-H "Content-Type: application/json" \
-d @predict_input.json
```

## Expected Response
```json
{
  "status": "success",
  "message": "Prediction successful",
  "data": {
    "model_type": "xgboost",
    "prediction": 0,
    "model_path": "models/xgboost_model_2025-10-30.joblib",
    "customer_id": "CUST001",
    "preprocessing_applied": true
  }
}
```
2. **GET** `/predict/{model_type}/customer/{customer_id}`
### Retrieve and predict churn for a specific customer from the database.

### Example curl request:
```bash
curl "http://127.0.0.1:8000/predict/neural-net/customer/CUST001"
```
### Expected Response:
```json
 {
  "status": "success",
  "message": "Prediction successful for customer CUST001",
  "data": {
    "model_type": "neural-net",
    "prediction": 0.0,
    "model_path": "models/nn_model_2025-10-29.pth",
    "customer_id": "CUST001",
    "feature_count": 50,
    "preprocessing_applied": false
  }
}
```
3. **GET** `/predict/{model_type}/batch/{batch_id}`
### Retrieve and predict churn for a batch of customers.

### Query Parameters:
| Parameter | Type | Description                                 |
| --------- | ---- | ------------------------------------------- |
| `limit`   | int  | Number of records to predict (default: 100) |
### Example curl request:
```bash
curl "http://127.0.0.1:8000/predict/xgboost/batch/BATCH001?limit=50"
```
### Expected Response:
```json
{
  "status": "success",
  "message": "Batch prediction successful for 50 customers",
  "data": {
    "model_type": "xgboost",
    "batch_id": "BATCH001",
    "predictions": [
      {"customer_id": "CUST001", "prediction": 0},
      {"customer_id": "CUST002", "prediction": 1}
    ],
    "model_path": "models/xgboost_model_2025-10-30.joblib",
    "prediction_count": 2,
    "preprocessing_applied": false
  }
}
```
### Error Responses
| Status Code | Description                       | Example                                                                                     |
| ----------- | --------------------------------- | ------------------------------------------------------------------------------------------- |
| `400`       | Invalid model type                | `"detail": "Invalid model_type. Must be one of ['xgboost', 'neural-net', 'random-forest']"` |
| `404`       | Model or customer/batch not found | `"detail": "customer_id: CUST001 not found"`                                                |
| `500`       | Preprocessing or prediction error | `"detail": "Failed to preprocess data: missing feature X"`                                  |
