import yaml
import pandas as pd

def load_preprocessing_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def validate_against_schema(df: pd.DataFrame, config_path: str = "config/config_preprocess.yaml"):
    """Check if DataFrame conforms to expected schema from preprocessing config."""
    config = load_preprocessing_config(config_path)
    issues = []

    expected_columns = config.get("columns", [])
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        issues.append({"missing_columns": missing_cols})

    return issues

def get_data_version():
    """Optional: return a version identifier (DVC, Git hash, etc.)"""
    return "v1"  # placeholder
