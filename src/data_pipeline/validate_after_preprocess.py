import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, Any
import os
from datetime import datetime
from src.data_pipeline.preprocess import DataPreprocessor


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
log_path = "src/data_pipeline/logs/preprocessed/validation.log"
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=log_path,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def validate_dataframe(df: pd.DataFrame, config_path: str, preprocessor: DataPreprocessor) -> pd.DataFrame:
    config = load_config(config_path)
    #logging = setup_logger(log_path, config["logging"]["log_level"])

    drop_cols = config.get("drop_columns", [])
    target_col = config.get("target_column", None)
    schema = config.get("schema", {})

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    drop_cols_lower = [c.lower() for c in drop_cols]
    df = df.drop(columns=[c for c in df.columns if c in drop_cols_lower], errors="ignore")
    columns = preprocessor.columns
    expected_cols_lower = [c.lower() for c in columns] 
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

     # Target column validation
    if target_col:
        target_col = target_col.strip().lower()
        if target_col not in df.columns:
            msg = f"Target column '{target_col}' not found in dataframe columns"
            logging.error(msg)
            raise ValueError(msg)

        if df[target_col].isnull().any():
            msg = f"Target column '{target_col}' contains null values"
            logging.error(msg)
            raise ValueError(msg)

        if df[target_col].nunique() < 2:
            msg = f"Target column '{target_col}' must have at least two unique values"
            logging.error(msg)
            raise ValueError(msg)

        # Imbalance warning
        value_counts = df[target_col].value_counts(normalize=True)
        if value_counts.max() > 0.95:
            logging.warning(
                f"Target column '{target_col}' is highly imbalanced: {value_counts.to_dict()}"
            )
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

    msg = f"Validation successful: {df.shape[0]} rows {df.shape[1]} columns"
    logging.info(msg)
    print(msg)
    return df
