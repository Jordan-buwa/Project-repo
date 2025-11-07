# Database API Documentation

## Overview

The **Database API** handles interactions with the PostgreSQL database used by the Predict API.  
It provides endpoints and utility functions to retrieve customer features and batch data for predictions.  

- **Logic location:** `src/api/routers/predict.py`  
- **Database connection utility:** `src/api/utils/database.py`  
- **Database type:** PostgreSQL  

---

## Database Connection

Connections are established using `psycopg2` via the helper function `get_db_connection()`:

```python
from src.api.utils.database import get_db_connection
```
- Returns a connection object compatible with Python `with` statements.
- Uses `RealDictCursor` for cursors to return rows as dictionaries.
- Handles connection lifecycle and cleanup automatically.

## Customer Data Table (customer_data)
The database contains a customer_data table with the following relevant columns:
| Column Name   | Data Type     | Description                                                 |
| ------------- | ------------- | ----------------------------------------------------------- |
| `customer_id` | string / UUID | Unique identifier for each customer.                        |
| `features`    | JSON / JSONB  | Dictionary of features for prediction.                      |
| `batch_id`    | string        | Identifier for batch operations.                            |
| `created_at`  | timestamp     | Timestamp for record creation, used to order batch queries. |

**Notes:**
- `features` must match the preprocessing schema expected by the model.
- `batch_id` and `created_at` allow retrieval of multiple records in batch mode.

### Endpoints Using the Database
### 1. Predict From Customer ID
Function: `predict_from_db_customer(model_type, customer_id)`
- Retrieves a single customer’s features from the database using `customer_id`.
- Loads the latest model of the specified `model_type`.
- Returns a prediction for that customer.
### SQL Query:
```sql
SELECT features 
FROM customer_data 
WHERE customer_id = %s;
```
### Error Handling:
- Raises `DataNotFoundError` if the `customer_id` does not exist.
- Raises preprocessing or model errors if prediction fails.

### Data Format
- Customer features: Stored as JSON objects. Example:
```json
{
  "revenue": 102.5,
  "mou": 189.0,
  "overage": 5.2,
  "roam": 1.0
  // include all other required features
}
```
- For batch queries, multiple such feature dictionaries are returned in the order of `created_at`.

### Example Usage
### Predict for a single customer:
```python
from src.api.routers.predict import predict_from_db_customer
response = predict_from_db_customer(model_type="xgboost", customer_id="C12345")
print(response.data)
```
### Predict for a batch:
```python
from src.api.routers.predict import predict_from_db_batch
response = predict_from_db_batch(model_type="xgboost", batch_id="BATCH01", limit=50)
print(response.data)
```