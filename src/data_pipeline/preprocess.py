#  Data Preprocessor Class
import os
import yaml
import pandas as pd
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine
from src.data_pipeline.ingest import DataIngestion
import numpy as np
from typing import List
import pickle
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, StandardScaler
load_dotenv()

#  Setup Logging


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


class DataPreprocessor:
    def __init__(self, config_path: str = "config/config_process.yaml", data_raw: pd.DataFrame = None):
        if data_raw is None:
            raise ValueError("data_raw (a pandas DataFrame) must be provided.")

        # Load config
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Use provided DataFrame
        self.df = data_raw.copy()

        # Config-based settings
        self.target_col = self.config["target_column"]
        self.num_cols = self.config["numerical_features"]
        self.cat_cols = self.config["categorical_features"]
        self.drop_col = self.config["drop_columns"]

        # For encoding and scaling
        self.label_encoders = {}
        self.scaler = None

        # Create folder for processed data & logging
        self.logger = setup_logger(
            self.config["logging"]["log_path"],
            self.config["logging"]["log_level"]
        )

    # Feature Engineering
    def combine_cols(self):
        """Create derived features safely (skip if raw columns missing)"""
        self.logger.info("Creating derived features...")

        def safe_addition(*cols):
            return sum(self.df[col] if col in self.df.columns else 0 for col in cols)

        def safe_divide(numerator, denominator):
            return numerator / (denominator + 1e-6)  # avoid divide by zero

        df = self.df

        if all(col in df.columns for col in ["outcalls", "incalls", "months"]):
            df["engagement_index"] = safe_divide(
                df["outcalls"] + df["incalls"], df["months"])
        if all(col in df.columns for col in ["models", "months"]):
            df["model_change_rate"] = safe_divide(df["models"], df["months"])
        if all(col in df.columns for col in ["overage", "revenue"]):
            df["overage_ratio"] = safe_divide(df["overage"], df["revenue"])
        if all(col in df.columns for col in ["mou", "mourec", "outcalls", "incalls", "peakvce", "opeakvce"]):
            df["call_activity_score"] = df["mou"] + df["mourec"] + 0.5 * \
                safe_addition("outcalls", "incalls", "peakvce", "opeakvce")
        if all(col in df.columns for col in ["dropvce", "blckvce", "unansvce", "dropblk"]):
            df["call_quality_issues"] = safe_addition(
                "dropvce", "blckvce", "unansvce", "dropblk")
        if all(col in df.columns for col in ["custcare", "retcalls", "retaccpt"]):
            df["cust_engagement_score"] = df["custcare"] + \
                df["retcalls"] + 2 * df["retaccpt"]
        if all(col in df.columns for col in ["overage", "directas", "recchrge"]):
            df["overuse_behavior"] = df["overage"] + \
                0.5 * safe_addition("directas", "recchrge")
        if all(col in df.columns for col in ["models", "eqpdays", "refurb"]):
            df["device_tenure_index"] = 0.5 * df["models"] + \
                (df["eqpdays"] / 100) + df["refurb"]
        if all(col in df.columns for col in ["age1", "age2", "children", "income"]):
            df["demographic_index"] = (
                (df["age1"] + df["age2"]) / 2) + df["children"] * 2 + (df["income"] / 10000)
        if all(col in df.columns for col in ["credita", "creditaa", "prizmub", "prizmtwn"]):
            df["socio_tier"] = df["credita"] + 2 * \
                df["creditaa"] + df["prizmub"] + 0.5 * df["prizmtwn"]
        if all(col in df.columns for col in ["occprof", "occcler", "occcrft", "occret", "occself"]):
            df["occupation_class"] = safe_addition(
                "occprof", "occcler", "occcrft", "occret", "occself")
        if all(col in df.columns for col in ["ownrent", "marryyes", "pcown", "creditcd", "travel", "truck", "rv"]):
            df["household_lifestyle_score"] = safe_addition(
                "ownrent", "marryyes", "pcown", "creditcd", "travel", "truck", "rv")
        if all(col in df.columns for col in ["changem", "changer", "newcelly", "newcelln", "refer"]):
            df["churn_change_score"] = safe_addition(
                "changem", "changer", "newcelly", "newcelln") - 0.5 * df["refer"]

        # Only add derived features that were successfully created
        created_features = [col for col in self.config.get(
            "combined_features", []) if col in df.columns]
        self.num_cols += created_features
        self.logger.info(f"Derived features added: {created_features}")

    def handle_missing_values(self):
        """Fill missing values for numerical and categorical"""
        self.logger.info("Handling missing values...")
        for col in self.num_cols:
            if self.df[col].isnull().any():
                self.df.fillna({col: self.df[col].median()}, inplace=True)

        for col in self.cat_cols:
            if self.df[col].isnull().any():
                fill_value = self.df[col].mode(
                )[0] if not self.df[col].mode().empty else "Unknown"
                self.df.fillna({col: fill_value}, inplace=True)

    def encode_categorical_variables(self):
        """Encode categorical features"""
        self.logger.info("Encoding categorical variables...")
        for col in self.cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le

    def encode_target_variable(self):
        """Encode target"""
        self.logger.info("Encoding target variable...")
        if self.df[self.target_col].dtype == "object":
            self.df[self.target_col] = self.df[self.target_col].map({
                "True": 1, "False": 0,
                "Yes": 1, "No": 0,
                "true": 1, "false": 0,
                "yes": 1, "no": 0
            }).fillna(0).astype(int)

    def feature_scaling(self):
        """Scale numerical features"""
        self.logger.info("Scaling numerical features...")
        self.scaler = StandardScaler()
        self.df[self.num_cols] = self.scaler.fit_transform(
            self.df[self.num_cols])

    def remove_unnecessary_columns(self):
        """Drop unnecessary raw columns, keeping derived features intact"""
        self.logger.info("Dropping unnecessary columns...")

        # Start with explicitly configured drop columns
        cols_to_drop = set(self.drop_col)

        # Automatically keep all derived features
        derived_features = set(self.config.get("combined_features", []))
        # Only drop columns that are not in derived features or target
        raw_cols_to_drop = set(self.df.columns) - \
            derived_features - {self.target_col}
        cols_to_drop.update(raw_cols_to_drop)

        # Drop columns safely
        for col in cols_to_drop:
            if col in self.df.columns:
                self.df.drop(columns=col, inplace=True)
                self.logger.info(f"Removed column: {col}")

        # Update numerical and categorical lists
        self.num_cols = [
            col for col in self.num_cols if col in self.df.columns]
        self.cat_cols = [
            col for col in self.cat_cols if col in self.df.columns]

    # Save Processed Data

    def save_preprocessed_data(self):
        """Save locally + PostgreSQL snapshot"""
        self.logger.info("Saving processed data...")

        # Local save
        local_path = "data/processed/processed_data.csv"
        self.df.to_csv(local_path, index=False)

        artifacts = {
            "label_encoders": {col: list(enc.classes_) for col, enc in self.label_encoders.items()},
            "scaler_params": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist()
            },
            "feature_names": self.df.columns.tolist(),
            "preprocessing_timestamp": datetime.utcnow().isoformat()
        }

        with open("src/data_pipeline/preprocessing_artifacts.json", "w") as f:
            json.dump(artifacts, f, indent=2)
        self.logger.info("Processed data saved locally.")
        # self.logger.info("Saving processed data to PostgreSQL...")
        # self.database_save()

    def database_save(self):
<<<<<<< HEAD
        # PostgreSQL snapshot (optional)
=======
        """Save processed data snapshot to PostgreSQL"""
        self.logger.info("Saving processed data snapshot to PostgreSQL...")
>>>>>>> upstream/main
        try:
            DB_USER = os.getenv("POSTGRES_USER", "jawpostgresdb")
            DB_PASS = os.getenv("POSTGRES_PASSWORD")
            DB_HOST = os.getenv(
                "POSTGRES_HOST", "jaw-postgresdb.postgres.database.azure.com")
            DB_PORT = os.getenv("POSTGRES_PORT", "5432")
            DB_NAME = os.getenv("POSTGRES_DB", "postgres")

            conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
            engine = create_engine(conn_str)

            snapshot_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            snapshot_table = f"processed_snapshot_{snapshot_id}"
            self.df.to_sql(snapshot_table, engine,
                           index=False, if_exists="replace")

            metadata = pd.DataFrame([{
                "snapshot_id": snapshot_id,
                "timestamp": datetime.utcnow(),
                "row_count": len(self.df),
                "feature_count": len(self.df.columns),
                "storage_type": "postgres",
                "artifact_file": "data/processed/preprocessing_artifacts.json"
            }])
            metadata.to_sql("preprocessing_metadata", engine,
                            index=False, if_exists="append")
            self.logger.info(
                f"Snapshot {snapshot_id} stored successfully in PostgreSQL.")
        except Exception as e:
            self.logger.error(
                f"Failed to save processed data to PostgreSQL: {e}")
            raise

    # Preprocessing Pipeline

    def run_preprocessing_pipeline(self):
        """Run the full preprocessing flow"""

        self.logger.info("Starting full preprocessing pipeline...")
        print("Starting full preprocessing pipeline...")
        self.combine_cols()
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.encode_target_variable()
        self.feature_scaling()
        self.remove_unnecessary_columns()
        self.save_preprocessed_data()

        if self.df.isnull().any().any():
            self.logger.error("NaN values detected after preprocessing!")
            raise ValueError(
                "Data contains NaNs after preprocessing pipeline.")

        self.logger.info("Preprocessing pipeline completed successfully.")
        print("Preprocessing pipeline completed successfully.")
        return self.df
    
class ProductionPreprocessor:
    """
    Inference-time preprocessor that loads saved artifacts
    and applies the same transformations as training.
    """
    
    def __init__(self, artifacts_path: str = "src/data_pipeline/preprocessing_artifacts.json"):
        """Load preprocessing artifacts from training"""
        self.logger = logging.getLogger(__name__)
        
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
        
        with open(artifacts_path, "r") as f:
            self.artifacts = json.load(f)
        
        # Reconstruct label encoders
        self.label_encoders = self.artifacts.get("label_encoders", {})
        
        # Reconstruct scaler
        self.scaler = StandardScaler()
        scaler_params = self.artifacts.get("scaler_params", {})
        self.scaler.mean_ = np.array(scaler_params.get("mean", []))
        self.scaler.scale_ = np.array(scaler_params.get("scale", []))
        
        # Feature metadata
        self.feature_names = self.artifacts.get("feature_names", [])
        self.num_cols = self.artifacts.get("numerical_features", [])
        self.cat_cols = self.artifacts.get("categorical_features", [])
        self.target_col = self.artifacts.get("target_column", "churn")
        self.drop_cols = self.artifacts.get("drop_columns", [])
        
        # Missing value fill strategies
        self.num_fill_values = self.artifacts.get("numerical_fill_values", {})
        self.cat_fill_values = self.artifacts.get("categorical_fill_values", {})
        
        self.logger.info(f"Loaded preprocessing artifacts from {artifacts_path}")
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applying the same preprocessing pipeline used in training.

        """
        df = data.copy()
        
        df = self._create_derived_features(df)
        
        df = self._remove_columns(df)
        
        df = self._handle_missing_values(df)
        
        df = self._encode_categorical(df)
        
        if self.target_col in df.columns:
            df = self._encode_target(df)
        
        df = self._scale_features(df)
        
        df = self._align_features(df)
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the same derived features as training"""
        if "outcalls" in df.columns and "incalls" in df.columns:
            df["engagement_index"] = (
                df["outcalls"] + df["incalls"]) / (df["months"] + 1)
        
        if "models" in df.columns and "months" in df.columns:
            df["model_change_rate"] = df["models"] / (df["months"] + 1)
        
        if "overage" in df.columns and "revenue" in df.columns:
            df["overage_ratio"] = df["overage"] / (df["revenue"] + 1)
        
        return df
    
    def _remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that were dropped during training"""
        cols_to_drop = [col for col in self.drop_cols if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using training statistics"""
        # Numerical features
        for col in self.num_cols:
            if col in df.columns and df[col].isnull().any():
                fill_value = self.num_fill_values.get(col, df[col].median())
                df[col].fillna(fill_value, inplace=True)
        
        # Categorical features
        for col in self.cat_cols:
            if col in df.columns and df[col].isnull().any():
                fill_value = self.cat_fill_values.get(col, "Unknown")
                df[col].fillna(fill_value, inplace=True)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using training encoders"""
        for col in self.cat_cols:
            if col in df.columns:
                # Get the classes from training
                classes = self.label_encoders.get(col, [])
                
                # Convert to string
                df[col] = df[col].astype(str)
                
                # Map to encoded values, handle unseen categories
                encoding_map = {cls: idx for idx, cls in enumerate(classes)}
                df[col] = df[col].map(encoding_map).fillna(-1).astype(int)
                
                # Log warning for unseen categories
                if (df[col] == -1).any():
                    self.logger.warning(
                        f"Unseen categories in column '{col}'. Encoded as -1."
                    )
        
        return df
    
    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode target variable if present"""
        if self.target_col in df.columns and df[self.target_col].dtype == "object":
            df[self.target_col] = df[self.target_col].map({
                "True": 1, "False": 0,
                "Yes": 1, "No": 0,
                "true": 1, "false": 0,
                "yes": 1, "no": 0
            }).fillna(0).astype(int)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using training scaler"""
        cols_to_scale = [col for col in self.num_cols if col in df.columns]
        
        if cols_to_scale and len(self.scaler.mean_) > 0:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure features match training feature order and presence"""
        # Remove target column if present (for inference)
        expected_features = [f for f in self.feature_names if f != self.target_col]
        
        # Add missing features with zeros
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0
                self.logger.warning(f"Missing feature '{feat}' - filled with 0")
        
        # Keep only expected features in the correct order
        df = df[expected_features]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return the expected feature names for model input"""
        return [f for f in self.feature_names if f != self.target_col]


def save_enhanced_preprocessing_artifacts(preprocessor_instance):
    """
    Enhanced artifact saving function to be called after preprocessing.
    Add this to your DataPreprocessor class or call it after run_preprocessing_pipeline.
    """
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Calculate fill values for missing data
    num_fill_values = {}
    for col in preprocessor_instance.num_cols:
        if col in preprocessor_instance.df.columns:
            num_fill_values[col] = float(preprocessor_instance.df[col].median())
    
    cat_fill_values = {}
    for col in preprocessor_instance.cat_cols:
        if col in preprocessor_instance.df.columns:
            mode_val = preprocessor_instance.df[col].mode()
            cat_fill_values[col] = mode_val[0] if not mode_val.empty else "Unknown"
    
    artifacts = {
        "label_encoders": {
            col: list(enc.classes_) 
            for col, enc in preprocessor_instance.label_encoders.items()
        },
        
        "scaler_params": {
            "mean": preprocessor_instance.scaler.mean_.tolist(),
            "scale": preprocessor_instance.scaler.scale_.tolist()
        },
        
        "feature_names": preprocessor_instance.df.columns.tolist(),
        "numerical_features": preprocessor_instance.num_cols,
        "categorical_features": preprocessor_instance.cat_cols,
        "target_column": preprocessor_instance.target_col,
        "drop_columns": preprocessor_instance.drop_col,
        
        "numerical_fill_values": num_fill_values,
        "categorical_fill_values": cat_fill_values,
        
        "preprocessing_timestamp": datetime.utcnow().isoformat(),
        "n_samples": len(preprocessor_instance.df),
        "n_features": len(preprocessor_instance.df.columns)
    }
    
    # Convert all numpy types to native Python types
    artifacts = convert_numpy_types(artifacts)
    
    # Save artifacts
    artifacts_path = "src/data_pipeline/preprocessing_artifacts.json"
    os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
    
    with open(artifacts_path, "w") as f:
        json.dump(artifacts, f, indent=2, default=str)  # Added default=str as extra safety
    
    # Also save a pickle backup for complex objects if needed
    pickle_path = "src/data_pipeline/preprocessing_artifacts.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "label_encoders": preprocessor_instance.label_encoders,
            "scaler": preprocessor_instance.scaler
        }, f)
    
    preprocessor_instance.logger.info(f"Enhanced artifacts saved to {artifacts_path}")
    return artifacts_path



if __name__ == "__main__":
    
    ingestion = DataIngestion("config/config_ingest.yaml")
    df_raw = ingestion.load_data()
    preprocessor = DataPreprocessor(
        config_path="config/config_process.yaml",
        data_raw=df_raw)
    preprocessor.run_preprocessing_pipeline()   
    save_enhanced_preprocessing_artifacts(preprocessor) 

    ProductionPreprocessor(
    artifacts_path="src/data_pipeline/preprocessing_artifacts.json"
    )

