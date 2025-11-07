# Example Requests and Responses

This document provides usage examples for the Predict, Metrics, and Data Validation APIs.

---

## 1. Predict API Example

### Endpoint:

```bash
POST /predict/{model_type}
```

### Example Request (using curl with JSON body)
)
```json
curl -X POST "http://127.0.0.1:8000/predict/xgboost" \
-H "Content-Type: application/json" \
-d '{
  "customer_data": [
    {
      "revenue": 102.5,
      "mou": 189.0,
      "overage": 5.2,
      "roam": 1.0
    },
    {
      "revenue": 85.3,
      "mou": 150.0,
      "overage": 2.1,
      "roam": 0.0
    }
  ]
}'
```
## Example Input JSON (CustomerData Schema)
```json
{
  "customer_data": [
    {
      "revenue": 102.5,
      "mou": 189.0,
      "overage": 5.2,
      "roam": 1.0
    },
    {
      "revenue": 85.3,
      "mou": 150.0,
      "overage": 2.1,
      "roam": 0.0
    }
  ]
}
```
## Example Response
```json
{
  "model_type": "xgboost",
  "prediction": [0, 1]
}
```
## 2. Metrics API Example
### Endpoint:

```http
POST /metrics/{model_type}
```

## Example Input JSON (test_input.json)
```json
[
  {
    "features": {
      "revenue": 200.3,
      "mou": 175.0,
      "overage": 6.4,
      "roam": 2.0
    },
    "target": 1
  },
  {
    "features": {
      "revenue": 50.2,
      "mou": 50.0,
      "overage": 1.2,
      "roam": 0.0
    },
    "target": 0
  }
]
```

## Example Response
```json
{
  "model_type": "xgboost",
  "model_path": "models/xgboost_model_2025-10-30.joblib",
  "metrics": {
    "accuracy": 0.86,
    "f1_score": 0.84,
    "roc_auc": 0.90
  }
}
```
## 3. Data Validation API Example
### Endpoint:
```http
POST /data_validation/validate
```
### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/data_validation/validate?csv_path=data/production/client.csv&model_type=xgboost&model_version=v1"
```
### Example Response (Success)
```json
{
  "timestamp": "2025-11-07T12:00:00.345Z",
  "rows": 100,
  "columns": 15,
  "issues": [],
  "stage": "Validated",
  "source": "csv",
  "model_type": "xgboost",
  "model_version": "v1"
}
```
### Example Response (With Issues)
```json
{
  "timestamp": "2025-11-07T12:03:21.210Z",
  "rows": 95,
  "columns": 12,
  "issues": [
    "Missing columns: {'roam', 'overage'}",
    "Column 'revenue' dtype mismatch: expected float64, got object"
  ],
  "stage": "Failed",
  "source": "database",
  "model_type": "random_forest",
  "model_version": "1.0"
}
```