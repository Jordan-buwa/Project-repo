import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, Any
import os
from datetime import datetime

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logger(log_path: str, log_level: str = "INFO"):
    base, ext = os.path.splitext(log_path)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path_ts = f"{base}_{timestamp}{ext}"

    os.makedirs(os.path.dirname(log_path_ts), exist_ok=True)
    logging.basicConfig(
        filename=log_path_ts,
        filemode="a", 
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    setup_logger(config["logging"]["log_path"], config["logging"]["log_level"])

    drop_cols = config.get("drop_columns", [])
    target_col = config.get("target_column", None)
    schema = config.get("schema", {})

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    drop_cols_lower = [c.lower() for c in drop_cols]
    df = df.drop(columns=[c for c in df.columns if c in drop_cols_lower], errors="ignore")

    expected_cols_lower = [c.lower() for c in schema.keys()] 
    missing_cols = [c for c in expected_cols_lower if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_cols_lower]

    if missing_cols:
        msg = f"Missing required columns: {missing_cols}"
        logging.error(msg)
        raise ValueError(msg)
    if extra_cols:
        msg = f"Unexpected columns found: {extra_cols}"
        logging.error(msg)
        raise ValueError(msg)

    for col in df.columns:
        col_schema = schema[col]
        col_dtype = col_schema.get("dtype")
        allow_null = col_schema.get("allow_null", True)
        allowed_values = col_schema.get("allowed_values", None)
        col_min = col_schema.get("min", None)
        col_max = col_schema.get("max", None)

        series = df[col]

        if not allow_null and series.isnull().any():
            msg = f"Column '{col}' contains null values but allow_null=False"
            logging.error(msg)
            raise ValueError(msg)

        if col_dtype == "float":
            if not np.issubdtype(series.dtype, np.number):
                msg = f"Column '{col}' type mismatch: expected float"
                logging.error(msg)
                raise TypeError(msg)
            if col_min is not None and (series < col_min).any():
                msg = f"Column '{col}' has values below min={col_min}"
                logging.error(msg)
                raise ValueError(msg)
            if col_max is not None and (series > col_max).any():
                msg = f"Column '{col}' has values above max={col_max}"
                logging.error(msg)
                raise ValueError(msg)

        elif col_dtype == "int":
            if not np.issubdtype(series.dtype, np.integer):
                msg = f"Column '{col}' type mismatch: expected int"
                logging.error(msg)
                raise TypeError(msg)
            if col_min is not None and (series < col_min).any():
                msg = f"Column '{col}' has values below min={col_min}"
                logging.error(msg)
                raise ValueError(msg)
            if col_max is not None and (series > col_max).any():
                msg = f"Column '{col}' has values above max={col_max}"
                logging.error(msg)
                raise ValueError(msg)
            if allowed_values is not None and not series.isin(allowed_values).all():
                msg = f"Column '{col}' contains values outside allowed: {allowed_values}"
                logging.error(msg)
                raise ValueError(msg)

        elif col_dtype == "category":
            if not pd.api.types.is_categorical_dtype(series) and not pd.api.types.is_object_dtype(series):
                msg = f"Column '{col}' type mismatch: expected category"
                logging.error(msg)
                raise TypeError(msg)
            if allowed_values is not None and not series.isin(allowed_values).all():
                msg = f"Column '{col}' contains values outside allowed: {allowed_values}"
                logging.error(msg)
                raise ValueError(msg)

        else:
            msg = f"Column '{col}' has unknown dtype '{col_dtype}' in schema"
            logging.error(msg)
            raise TypeError(msg)

    df = df[[c for c in expected_cols_lower if c in df.columns]]

    msg = f"Validation successful: {df.shape[0]} rows × {df.shape[1]} columns"
    logging.info(msg)
    print(msg)
    return df
