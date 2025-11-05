# Example Requests and Responses

---

##  Predict API Example

### Example Request (using curl)
```bash
curl -X POST "http://127.0.0.1:8000/predict/xgboost"
```
## Example Input JSON (predict_input.json)
```json
{
  "features": {
    "revenue": 102.5,
    "mou": 189.0,
    "overage": 5.2,
    "roam": 1.0
  }
}
```

## Example Response
```json
{
  "model_type": "xgboost",
  "prediction": [0, 1, 1, 0]
}
```
## Metrics API Example
### Example Request (using curl)
curl -X POST "http://127.0.0.1:8000/metrics/xgboost"

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