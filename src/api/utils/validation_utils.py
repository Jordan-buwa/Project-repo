import os
import pandas as pd
import yaml
import time
import psycopg2
import json
from typing import Tuple, List, Dict, Optional

def load_config(config_path: str):
    """Load validation config from YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Validation config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model_schema_config(config_path: str, model_type: str) -> Dict:
    """Get schema configuration for a specific model type from config."""
    try:
        if not os.path.exists(config_path):
            return {}
            
        config = load_config(config_path)
        
        # Checking if model has specific schema configuration
        if ('registered_models' in config and 
            model_type in config['registered_models'] and
            'schema_path' in config['registered_models'][model_type]):
            
            model_config = config['registered_models'][model_type]
            schema_config = {
                'schema_path': model_config.get('schema_path'),
                'default_version': model_config.get('default_version'),
                'base_path': config.get('schema_config', {}).get('base_path', 'models/'),
                'naming_patterns': config.get('schema_config', {}).get('naming_patterns', [
                    "{version}_schema.json",
                    "v{version}_schema.json", 
                    "{version}.json"
                ])
            }
            return schema_config
        
        # Fallback to global schema configuration
        schema_config = config.get('schema_config', {})
        return {
            'schema_path': os.path.join(
                schema_config.get('base_path', 'models/'),
                model_type,
                schema_config.get('default_location', 'schemas/')
            ),
            'default_version': None,
            'base_path': schema_config.get('base_path', 'models/'),
            'naming_patterns': schema_config.get('naming_patterns', [
                "{version}_schema.json",
                "v{version}_schema.json",
                "{version}.json"
            ])
        }
        
    except Exception as e:
        print(f"Error getting schema config for {model_type}: {e}")
        return {}

def get_allowed_model_types(config_path: str, model_dir: str = "models/") -> List[str]:
    """Get allowed model types from config file or model directory."""
    try:
        if os.path.exists(config_path):
            config = load_config(config_path)
            
            # model_selection.choices
            if ('model_selection' in config and 
                config['model_selection'] is not None and 
                'choices' in config['model_selection'] and 
                config['model_selection']['choices'] is not None):
                return config['model_selection']['choices']
            
            # registered_models keys
            if 'registered_models' in config and config['registered_models'] is not None:
                return list(config['registered_models'].keys())
            
            # Fallback to default models
            return ["random_forest", "xgboost", "neural_network"]
        
        # Fallback: get model types from model directory
        if os.path.exists(model_dir):
            model_types = [name for name in os.listdir(model_dir) 
                          if os.path.isdir(os.path.join(model_dir, name))]
            return model_types
        
        return ["random_forest", "xgboost", "neural_network"]
        
    except Exception as e:
        print(f"Error getting allowed model types: {e}")
        return ["random_forest", "xgboost", "neural_network"]

def find_schema_file(schema_dir: str, model_version: str, naming_patterns: List[str]) -> Optional[str]:
    """Find schema file using multiple naming patterns."""
    for pattern in naming_patterns:
        schema_filename = pattern.format(version=model_version)
        potential_path = os.path.join(schema_dir, schema_filename)
        if os.path.exists(potential_path):
            return potential_path
    return None

def validate_schema(df: pd.DataFrame, model_type: str, model_version: str, model_dir: str = "models/", config_path: str = None):
    """
    Validate DataFrame against schema using configuration-based schema discovery.
    """
    try:
        print(f"validate_schema called with model_dir: {model_dir}")  # Debug
        print(f"validate_schema called with config_path: {config_path}")  # Debug
        
        # using configuration-based schema discovery if config_path is provided
        if config_path and os.path.exists(config_path):
            schema_config = get_model_schema_config(config_path, model_type)
            schema_dir = schema_config.get('schema_path', os.path.join(model_dir, model_type, "schemas"))
            naming_patterns = schema_config.get('naming_patterns', [
                "{version}_schema.json",
                "v{version}_schema.json",
                "{version}.json"
            ])
            
            print(f"Using schema_dir from config: {schema_dir}")  # Debug
        else:
            # Fallback to directory-based discovery
            schema_dir = os.path.join(model_dir, model_type, "schemas")
            naming_patterns = [
                "{version}_schema.json",
                "v{version}_schema.json", 
                "{version}.json"
            ]
            print(f"Using fallback schema_dir: {schema_dir}")  # Debug
        
        # Ensuring schema_dir is absolute path
        if not os.path.isabs(schema_dir):
            schema_dir = os.path.join(os.getcwd(), schema_dir)
        
        print(f"Final schema_dir: {schema_dir}")  # Debug
        print(f"Schema dir exists: {os.path.exists(schema_dir)}")  # Debug
        
        if os.path.exists(schema_dir):
            print(f"Files in schema dir: {os.listdir(schema_dir)}")  # Debug
        
        schema_file = find_schema_file(schema_dir, model_version, naming_patterns)
        
        if not schema_file:
            available_files = []
            if os.path.exists(schema_dir):
                available_files = os.listdir(schema_dir)
            raise FileNotFoundError(
                f"Schema file not found for {model_type} v{model_version} in {schema_dir}. "
                f"Available files: {available_files}. "
                f"Tried patterns: {naming_patterns}"
            )
        
        print(f"Found schema file: {schema_file}")  # Debug
        
        with open(schema_file, "r") as f:
            schema = json.load(f)
        
        issues = []
        expected_cols = schema.get("required_columns", [])
        dtypes = schema.get("dtypes", {})
        
        # Checking columns
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Checking dtypes
        for col in expected_cols:
            if col in df.columns and col in dtypes:
                expected_dtype = dtypes[col]
                actual_dtype = str(df[col].dtype)
                
                dtype_mapping = {
                    'integer': 'int64', 'bigint': 'int64', 'smallint': 'int64',
                    'numeric': 'float64', 'real': 'float64', 'double precision': 'float64',
                    'text': 'object', 'varchar': 'object', 'char': 'object',
                    'boolean': 'bool', 'date': 'object', 'timestamp': 'datetime64[ns]',
                    'timestamptz': 'datetime64[ns]'
                }
                
                normalized_expected = dtype_mapping.get(expected_dtype.lower(), expected_dtype)
                
                if normalized_expected != actual_dtype:
                    issues.append(f"Column '{col}' dtype mismatch: expected {normalized_expected}, got {actual_dtype}")
        
        return issues
        
    except Exception as e:
        raise Exception(f"Schema validation failed for {model_type} v{model_version}: {str(e)}")

def fetch_data(db_connection_string: str = None,
               query: str = None,
               csv_path: str = "data/production/client.csv",
               db_delay_seconds: int = 600,
               max_rows: int = 100) -> Tuple[pd.DataFrame, str]:
    """
    Fetch data from database or CSV. Returns (df, source).
    db_delay_seconds is the minimum time to wait between database queries; 
    fallback to CSV if database fails.
    """
    df = None
    source = None
    now = time.time()
    
    # trying database first
    if db_connection_string and query:
        try:
            # PostgreSQL connection using psycopg2
            conn = psycopg2.connect(db_connection_string)
            
            # Adding LIMIT clause to query for PostgreSQL syntax
            if "LIMIT" not in query.upper():
                query_with_limit = f"{query} LIMIT {max_rows}"
            else:
                query_with_limit = query
                
            df = pd.read_sql_query(query_with_limit, conn)
            conn.close()
            source = "database"
        except Exception as e:
            print(f"PostgreSQL fetch failed: {e}. Falling back to CSV.")
    
    # fallback to CSV if DB failed or not configured
    if df is None or df.empty:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Limitting rows for CSV as well to match max_rows parameter
        if len(df) > max_rows:
            df = df.head(max_rows)
        source = "csv"
    
    return df, source