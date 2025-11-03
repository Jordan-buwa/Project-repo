import yaml
import os
import subprocess
import pandas as pd

CONFIG_PATH = "config/config_preprocess.yaml"

def load_config(config_path=CONFIG_PATH):
    """Load preprocessing/validation config from YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_data_version(file_path="data/processed/preprocessed.csv"):
    """Return data version using DVC hash if available, else git commit."""
    try:
        return subprocess.getoutput(f"dvc hash {file_path}")
    except Exception:
        return subprocess.getoutput("git rev-parse HEAD")

def validate_against_schema(df: pd.DataFrame):
    """Validate DataFrame against schema defined in preprocessing config."""
    config = load_config()
    schema = config.get("schema", {})
    issues = []

    for col, rules in schema.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
            continue

        # Checking dtype
        expected_dtype = rules.get("dtype")
        if expected_dtype:
            if expected_dtype == "float" and not pd.api.types.is_float_dtype(df[col]):
                issues.append(f"Column '{col}' expected float but got {df[col].dtype}")
            elif expected_dtype == "int" and not pd.api.types.is_integer_dtype(df[col]):
                issues.append(f"Column '{col}' expected int but got {df[col].dtype}")
            elif expected_dtype == "str" and not pd.api.types.is_string_dtype(df[col]):
                issues.append(f"Column '{col}' expected str but got {df[col].dtype}")

        # Checking nulls
        if not rules.get("allow_null", True) and df[col].isna().any():
            issues.append(f"Column '{col}' contains nulls but allow_null=False")

        # Checking allowed values
        allowed_values = rules.get("allowed_values")
        if allowed_values is not None and not df[col].dropna().isin(allowed_values).all():
            issues.append(f"Column '{col}' contains values outside allowed set")

    return issues
