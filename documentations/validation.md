# Data Validation API Documentation

## Overview

The **Data Validation API** validates datasets used for model training, testing, or production scoring.  
It supports both **database** and **CSV** data sources, comparing the dataset’s schema against the versioned model schema stored under each model’s directory.

- **API script:** `src/api/routers/validate.py`  
- **Validation logic:** `src/api/utils/validation_utils.py`  
- **Configuration:** `.env` and `config/config_api_data-val.yaml`  
- **Cache file:** `src/api/cache/data_validation_cache.json`  

---

## Endpoints Summary

| Method | Endpoint | Description |
|--------|-----------|-------------|
| **POST** | `/data_validation/validate` | Fetch and validate a dataset against a registered model schema. |

---

## Request Parameters

| Parameter | Type | Required | Description |
|------------|------|-----------|--------------|
| `db_connection_string` | `string` | false | PostgreSQL connection string (e.g., `"postgresql://user:pass@host:port/db"`). |
| `query` | `string` | false | SQL query to fetch data from the database. |
| `csv_path` | `string` | true | Local CSV path if DB connection is not used. Default: `"data/production/client.csv"`. |
| `db_delay_seconds` | `int` | false | Minimum delay between database queries. Default: `600`. |
| `max_rows` | `int` | false | Maximum rows to fetch from data source. Default: `100`. |
| `model_type` | `string` | true | Type of model whose schema to validate against (e.g., `"xgboost"`, `"random_forest"`, `"neural_net"`). |
| `model_version` | `string` | true | Version of the model schema to use for validation (e.g., `"v1"`, `"1.0"`). |

---

## Example Requests

### Validate Using Local CSV
```bash
curl -X POST "http://127.0.0.1:8000/data_validation/validate?csv_path=data/production/client.csv&model_type=xgboost&model_version=v1"
```
### Validate Using Database
```bash
curl -X POST "http://127.0.0.1:8000/data_validation/validate?db_connection_string=postgresql://user:password@localhost:5432/mydb&query=SELECT * FROM clients&model_type=random_forest&model_version=1.0"
```
## Validation Logic

### 1. Load Configuration
The API reloads environment variables on each request (via `dotenv`) and uses paths defined in:

- `DATA_VALIDATION_CACHE_FILE` → stores validation history  
- `VALIDATION_CONFIG` → YAML defining model schema paths and naming patterns  
- `MODEL_DIR` → root directory containing versioned model schemas  

### 2. Discover Allowed Model Types
Allowed model types are determined from either:
- YAML keys under `registered_models`, or  
- folders in the `/models/` directory  

### 3. Fetch Data
The API retrieves data from:
- **PostgreSQL**, if `db_connection_string` and `query` are provided, or  
- **Local CSV** file at `csv_path`  

The dataset is limited to `max_rows` entries.

### 4. Schema Validation
For each model type:
- Loads the expected schema file (e.g., `models/xgboost/schemas/v1_schema.json`)  
- Compares required columns and data types  
- Reports missing columns or dtype mismatches  

### 5. Cache Results
Validation outcomes (timestamp, model type, issues, and data source) are stored in:  
`src/api/cache/data_validation_cache.json`

## Developer Notes

- The API reloads environment variables on every request to ensure fresh configuration.
- Validation relies on YAML definitions in `config/config_api_data-val.yaml`.
- Schema discovery supports multiple naming conventions (e.g., `{version}_schema.json`, `v{version}_schema.json`).
- If database fetch fails, the service automatically falls back to CSV data.
- Cached results can be cleared manually by deleting `src/api/cache/data_validation_cache.json`.
- To add a new model type, update the `registered_models` section in the YAML config or create a new folder under `/models/`.

### Example Successful Response
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
### Example Response with Issues
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
### Error Responses
| Status Code | Description              | Example                                                                                     |
| ----------- | ------------------------ | ------------------------------------------------------------------------------------------- |
| `400`       | Invalid `model_type`     | `"detail": "Invalid model_type. Must be one of ['xgboost', 'random_forest', 'neural_net']"` |
| `404`       | Schema or CSV not found  | `"detail": "Schema file not found for xgboost v1 in models/xgboost/schemas/"`               |
| `500`       | Validation or DB failure | `"detail": "Validation failed: PostgreSQL fetch failed: connection refused"`                |
### Example Schema File (`v1_schema.json`)
```json
{
  "required_columns": ["revenue", "mou", "overage", "roam"],
  "dtypes": {
    "revenue": "numeric",
    "mou": "integer",
    "overage": "numeric",
    "roam": "boolean"
  }
}
