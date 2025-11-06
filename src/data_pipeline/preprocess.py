#  Data Preprocessor Class
import os
import yaml
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.data_pipeline.ingest import DataIngestion
load_dotenv()

#  Setup Logging


def setup_logger(log_path: str, log_level: str = "INFO"):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path + f"preprocess_{timestamp}.log",
        filemode="a",
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_path: str = "config/config_process.yaml", data_raw: pd.DataFrame = None):

        # Load config
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        if data_raw is not None:
            # Use provided DataFrame
            self.df = data_raw.copy()
        else: 
            try: self.df = pd.read_csv("data/backup/ingested.csv")
            except: raise ValueError("data must be provided")

        # Config-based settings
        self.target_col = self.config["target_column"]
        self.num_cols = self.config["numerical_features"]
        self.cat_cols = self.config["categorical_features"]
        self.drop_col = self.config["must_drop_columns"]
        self.features_to_drop = self.config["features_to_drop"]
        self.derived_features_config = self.config.get("combined_features", [])

        # For encoding and scaling
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Store feature engineering parameters
        self.feature_engineering_params = {}
        self.numerical_fill_values = {}
        self.categorical_fill_values = {}
        self.target_mapping = {}

        # Create folder for processed data & logging
        self.logger = setup_logger(
            self.config["logging"]["log_path"],
            self.config["logging"]["log_level"]
        )

    def handle_missing_values(self):
        """Fill missing values and store fill strategies - MUST BE DONE BEFORE FEATURE ENGINEERING"""
        self.logger.info("Handling missing values...")
        
        # Numerical features
        for col in self.num_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                fill_value = self.df[col].median()
                self.df[col].fillna(fill_value, inplace=True)
                self.numerical_fill_values[col] = float(fill_value)
                self.logger.info(f"Filled missing values in {col} with median: {fill_value}")

        # Categorical features
        for col in self.cat_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                if not self.df[col].mode().empty:
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = "Unknown"
                self.df[col].fillna(fill_value, inplace=True)
                self.categorical_fill_values[col] = fill_value
                self.logger.info(f"Filled missing values in {col} with mode: {fill_value}")

        # Handle missing values in potential source columns for derived features
        
        for col in self.features_to_drop:
            if col in self.df.columns and self.df[col].isnull().any():
                if col in self.num_cols:
                    fill_value = self.df[col].median()
                    self.df[col].fillna(fill_value, inplace=True)
                    if col not in self.numerical_fill_values:
                        self.numerical_fill_values[col] = float(fill_value)
                elif col in self.cat_cols:
                    if not self.df[col].mode().empty:
                        fill_value = self.df[col].mode()[0]
                    else:
                        fill_value = "Unknown"
                    self.df[col].fillna(fill_value, inplace=True)
                    if col not in self.categorical_fill_values:
                        self.categorical_fill_values[col] = fill_value

    def _safe_feature_creation(self, feature_name: str, creation_func, required_cols: list):
        """Safely create a feature after missing values are handled"""
        if all(col in self.df.columns for col in required_cols):
            try:
                # Check if all required columns have no missing values
                if self.df[required_cols].isnull().any().any():
                    missing_cols = self.df[required_cols].columns[self.df[required_cols].isnull().any()].tolist()
                    self.logger.warning(f"Cannot create {feature_name}: missing values in {missing_cols}")
                    return None
                
                result = creation_func()
                
                # Store creation parameters for production use
                self.feature_engineering_params[feature_name] = {
                    "required_columns": required_cols,
                    "operation": "custom",
                    "created_at": datetime.utcnow().isoformat()
                }
                
                self.logger.info(f"Successfully created feature: {feature_name}")
                return result
            except Exception as e:
                self.logger.warning(f"Failed to create feature {feature_name}: {str(e)}")
                return None
        else:
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            self.logger.warning(f"Cannot create {feature_name}: missing columns {missing_cols}")
            return None
    def combine_cols(self):
        """Create derived features AFTER missing values are handled"""
        self.logger.info("Creating derived features after missing value handling...")
        
        created_features = []
        
        # Engagement Index - (outcalls + incalls) / (months + 1)
        eng_result = self._safe_feature_creation(
            "engagement_index",
            lambda: (self.df["outcalls"] + self.df["incalls"]) / (self.df["months"] + 1),
            ["outcalls", "incalls", "months"]
        )
        if eng_result is not None:
            self.df["engagement_index"] = eng_result
            created_features.append("engagement_index")

        # Model Change Rate - models / (months + 1)
        model_result = self._safe_feature_creation(
            "model_change_rate",
            lambda: self.df["models"] / (self.df["months"] + 1),
            ["models", "months"]
        )
        if model_result is not None:
            self.df["model_change_rate"] = model_result
            created_features.append("model_change_rate")

        # Overage Ratio - overage / (revenue + 1)
        overage_result = self._safe_feature_creation(
            "overage_ratio",
            lambda: self.df["overage"] / (self.df["revenue"] + 1),
            ["overage", "revenue"]
        )
        if overage_result is not None:
            self.df["overage_ratio"] = overage_result
            created_features.append("overage_ratio")

        # Call Activity Score - mou + mourec + 0.5*(outcalls + incalls + peakvce + opeakvce)
        call_activity_result = self._safe_feature_creation(
            "call_activity_score",
            lambda: (self.df["mou"] + self.df["mourec"] + 
                    0.5 * (self.df["outcalls"] + self.df["incalls"] + 
                          self.df["peakvce"] + self.df["opeakvce"])),
            ["mou", "mourec", "outcalls", "incalls", "peakvce", "opeakvce"]
        )
        if call_activity_result is not None:
            self.df["call_activity_score"] = call_activity_result
            created_features.append("call_activity_score")

        # Call Quality Issues - dropvce + blckvce + unansvce + dropblk
        call_quality_result = self._safe_feature_creation(
            "call_quality_issues",
            lambda: (self.df["dropvce"] + self.df["blckvce"] + 
                    self.df["unansvce"] + self.df["dropblk"]),
            ["dropvce", "blckvce", "unansvce", "dropblk"]
        )
        if call_quality_result is not None:
            self.df["call_quality_issues"] = call_quality_result
            created_features.append("call_quality_issues")

        # Customer Engagement Score - custcare + retcalls + 2*retaccpt
        cust_engagement_result = self._safe_feature_creation(
            "cust_engagement_score",
            lambda: (self.df["custcare"] + self.df["retcalls"] + 
                    2 * self.df["retaccpt"]),
            ["custcare", "retcalls", "retaccpt"]
        )
        if cust_engagement_result is not None:
            self.df["cust_engagement_score"] = cust_engagement_result
            created_features.append("cust_engagement_score")

        # Overuse Behavior - overage + 0.5*(directas + recchrge)
        overuse_result = self._safe_feature_creation(
            "overuse_behavior",
            lambda: (self.df["overage"] + 
                    0.5 * (self.df["directas"] + self.df["recchrge"])),
            ["overage", "directas", "recchrge"]
        )
        if overuse_result is not None:
            self.df["overuse_behavior"] = overuse_result
            created_features.append("overuse_behavior")

        # Device Tenure Index - 0.5*models + eqpdays/100 + refurb
        device_tenure_result = self._safe_feature_creation(
            "device_tenure_index",
            lambda: (0.5 * self.df["models"] + 
                    self.df["eqpdays"] / 100 + self.df["refurb"]),
            ["models", "eqpdays", "refurb"]
        )
        if device_tenure_result is not None:
            self.df["device_tenure_index"] = device_tenure_result
            created_features.append("device_tenure_index")

        # Demographic Index - ((age1 + age2)/2) + children*2 + income/10000
        demographic_result = self._safe_feature_creation(
            "demographic_index",
            lambda: ((self.df["age1"] + self.df["age2"]) / 2 + 
                    self.df["children"] * 2 + self.df["income"] / 10000),
            ["age1", "age2", "children", "income"]
        )
        if demographic_result is not None:
            self.df["demographic_index"] = demographic_result
            created_features.append("demographic_index")

        # Update numerical features list with newly created features
        self.num_cols = list(set(self.num_cols + created_features))
        self.logger.info(f"Derived features created: {created_features}")
        

    def encode_categorical_variables(self):
        """Encode categorical features"""
        self.logger.info("Encoding categorical variables...")
        for col in self.cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            self.logger.info(f"Encoded categorical variable: {col}")

    def encode_target_variable(self):
        """Encode target variable consistently"""
        self.logger.info("Encoding target variable...")
        if self.target_col in self.df.columns:
            if self.df[self.target_col].dtype == "object":
                # Create consistent mapping
                self.target_mapping = {
                    "True": 1, "False": 0,
                    "Yes": 1, "No": 0,
                    "true": 1, "false": 0,
                    "yes": 1, "no": 0,
                    "1": 1, "0": 0
                }
                self.df[self.target_col] = (
                    self.df[self.target_col]
                    .astype(str)
                    .str.strip()
                    .map(self.target_mapping)
                    .fillna(0)
                    .astype(int)
                )
                self.logger.info(f"Encoded target variable {self.target_col}")
            else:
                # Ensure it's integer type
                self.df[self.target_col] = self.df[self.target_col].astype(int)

    def feature_scaling(self):
        """Scale numerical features and store scaler"""
        self.logger.info("Scaling numerical features...")
        numerical_cols_to_scale = [col for col in self.num_cols if col in self.df.columns]
        
        if numerical_cols_to_scale:
            self.df[numerical_cols_to_scale] = self.scaler.fit_transform(
                self.df[numerical_cols_to_scale]
            )
            self.logger.info(f"Scaled numerical features: {numerical_cols_to_scale}")

    def remove_unnecessary_columns(self):
        """Drop unnecessary raw columns, keeping derived features intact"""
        self.logger.info("Dropping unnecessary columns...")

        # Start with explicitly configured drop columns
        cols_to_drop = set(self.drop_col)|set(self.features_to_drop)

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
        self.logger.info("Processed data saved locally.")
        self.logger.info("Saving processed data to PostgreSQL...")
        self.database_save()

    def database_save(self):
        """Save processed data snapshot to PostgreSQL"""
        self.logger.info("Saving processed data snapshot to PostgreSQL...")
        try:
            DB_USER = os.getenv("POSTGRES_DB_USER", "jawpostgresdb")
            DB_PASS = os.getenv("POSTGRES_PASSWORD")
            DB_HOST = os.getenv(
                "POSTGRES_HOST", "jaw-postgresdb.postgres.database.azure.com")
            DB_PORT = os.getenv("POSTGRES_PORT", "5432")
            DB_NAME = os.getenv("POSTGRES_DB_NAME", "postgres")

            conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
            engine = create_engine(conn_str)

            snapshot_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            snapshot_table = f"customer_prod_data"
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

    def run_preprocessing_pipeline(self):
        """Run the full preprocessing flow in correct order"""
        self.logger.info("Starting full preprocessing pipeline...")
        print("Starting full preprocessing pipeline...")
        
        try:
            # CORRECT ORDER: Handle missing values FIRST
            self.handle_missing_values()
            
            # THEN create derived features using clean data
            self.combine_cols()
            
            # Continue with other preprocessing steps
            self.encode_categorical_variables()
            self.encode_target_variable()
            self.feature_scaling()
            self.remove_unnecessary_columns()
            
            # Validate no NaN values remain
            if self.df.isnull().any().any():
                nan_cols = self.df.columns[self.df.isnull().any()].tolist()
                self.logger.error(f"NaN values detected in columns: {nan_cols}")
                raise ValueError(f"Data contains NaNs after preprocessing in: {nan_cols}")
            
            self.save_preprocessed_data()
            _ = self.get_feature_names()
            self.logger.info("Preprocessing pipeline completed successfully.")
            print("Preprocessing pipeline completed successfully.")
            return self.df
            
        except Exception as e:
            self.logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise

    def get_feature_names(self) -> list[str]:
        """Return the expected feature names for model input"""
        self.columns = self.df.columns
        return [col for col in self.columns if col != self.target_col]



class ProductionPreprocessor(DataPreprocessor):
    """
    Inference-time preprocessor that inherits from DataPreprocessor
    and loads artifacts to replicate training transformations
    """
    
    def __init__(self, artifacts_path: str = "src/data_pipeline/preprocessing_artifacts.json"):
        """Initialize from saved artifacts instead of config and raw data"""
        self.artifacts_path = artifacts_path
        
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
        
        # Load artifacts
        with open(artifacts_path, "r") as f:
            self.artifacts = json.load(f)
        
        # Initialize parent without config and data (we'll set everything from artifacts)
        self._initialize_from_artifacts()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Production preprocessor initialized from {artifacts_path}")

    def _initialize_from_artifacts(self):
        """Initialize all attributes from saved artifacts"""
        # Load basic configuration from artifacts
        self.feature_names = self.artifacts.get("feature_names", [])
        self.num_cols = self.artifacts.get("numerical_features", [])
        self.cat_cols = self.artifacts.get("categorical_features", [])
        self.target_col = self.artifacts.get("target_column", "churn")
        self.drop_col = self.artifacts.get("drop_columns", [])
        
        # Load transformation parameters
        self.feature_engineering_params = self.artifacts.get("feature_engineering_params", {})
        self.numerical_fill_values = self.artifacts.get("numerical_fill_values", {})
        self.categorical_fill_values = self.artifacts.get("categorical_fill_values", {})
        self.target_mapping = self.artifacts.get("target_mapping", {})
        
        # Reconstruct transformation objects
        self.label_encoders = self._reconstruct_label_encoders()
        self.scaler = self._reconstruct_scaler()
        
        # Initialize empty dataframe (will be set during preprocessing)
        self.df = None

    def _reconstruct_label_encoders(self):
        """Reconstruct LabelEncoder objects from stored classes"""
        encoders = {}
        label_encoders_data = self.artifacts.get("label_encoders", {})
        
        for col, classes in label_encoders_data.items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            encoders[col] = le
            
        return encoders

    def _reconstruct_scaler(self):
        """Reconstruct StandardScaler from stored parameters"""
        scaler = StandardScaler()
        scaler_params = self.artifacts.get("scaler_params", {})
        
        if scaler_params:
            scaler.mean_ = np.array(scaler_params.get("mean", []))
            scaler.scale_ = np.array(scaler_params.get("scale", []))
            scaler.var_ = np.array(scaler_params.get("var", []))
            scaler.n_samples_seen_ = scaler_params.get("n_samples_seen", 0)
            
        return scaler

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the exact same preprocessing pipeline used in training
        Inherits the same method structure but uses stored parameters
        """
        self.df = data.copy()
        
        try:
            # Step 1: Handle missing values FIRST (using training statistics)
            self._handle_missing_values_production()
            
            # Step 2: Create derived features AFTER missing values are handled
            self._combine_cols_production()
            
            # Step 3: Encode categorical variables (using training encoders)
            self._encode_categorical_variables_production()
            
            # Step 4: Scale numerical features (using training scaler)
            self._feature_scaling_production()
            
            # Step 5: Align features with training schema
            self.df = self._align_features(self.df)
            
            self.logger.info("Production preprocessing completed successfully.")
            return self.df
            
        except Exception as e:
            self.logger.error(f"Production preprocessing failed: {str(e)}")
            raise

    def _handle_missing_values_production(self):
        """Fill missing values using training statistics - FIRST STEP"""
        self.logger.info("Handling missing values in production...")
        
        # Numerical features
        for col, fill_value in self.numerical_fill_values.items():
            if col in self.df.columns and self.df[col].isnull().any():
                self.df[col].fillna(fill_value, inplace=True)
        
        # Categorical features
        for col, fill_value in self.categorical_fill_values.items():
            if col in self.df.columns and self.df[col].isnull().any():
                self.df[col].fillna(fill_value, inplace=True)
        
        # Handle potential source columns for derived features
        potential_derived_source_cols = self.artifacts.get("potential_derived_source_cols", [])
        
        for col in potential_derived_source_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                if col in self.numerical_fill_values:
                    self.df[col].fillna(self.numerical_fill_values[col], inplace=True)
                elif col in self.categorical_fill_values:
                    self.df[col].fillna(self.categorical_fill_values[col], inplace=True)
                else:
                    # Fallback: use median for numerical, mode for categorical
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        if not self.df[col].mode().empty:
                            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                        else:
                            self.df[col].fillna("Unknown", inplace=True)

    def _combine_cols_production(self):
        """Create derived features AFTER missing values are handled"""
        self.logger.info("Creating derived features in production...")
        
        feature_operations = {
            "engagement_index": lambda: (self.df["outcalls"] + self.df["incalls"]) / (self.df["months"] + 1),
            "model_change_rate": lambda: self.df["models"] / (self.df["months"] + 1),
            "overage_ratio": lambda: self.df["overage"] / (self.df["revenue"] + 1),
            "call_activity_score": lambda: (self.df["mou"] + self.df["mourec"] + 
                                          0.5 * (self.df["outcalls"] + self.df["incalls"] + 
                                                self.df["peakvce"] + self.df["opeakvce"])),
            "call_quality_issues": lambda: (self.df["dropvce"] + self.df["blckvce"] + 
                                          self.df["unansvce"] + self.df["dropblk"]),
            "cust_engagement_score": lambda: (self.df["custcare"] + self.df["retcalls"] + 
                                            2 * self.df["retaccpt"]),
            "overuse_behavior": lambda: (self.df["overage"] + 
                                       0.5 * (self.df["directas"] + self.df["recchrge"])),
            "device_tenure_index": lambda: (0.5 * self.df["models"] + 
                                          self.df["eqpdays"] / 100 + self.df["refurb"]),
            "demographic_index": lambda: ((self.df["age1"] + self.df["age2"]) / 2 + 
                                        self.df["children"] * 2 + self.df["income"] / 10000)
        }
        
        for feature_name, operation in feature_operations.items():
            if feature_name in self.feature_engineering_params:
                required_cols = self.feature_engineering_params[feature_name].get("required_columns", [])
                if all(col in self.df.columns for col in required_cols):
                    try:
                        # Ensure no missing values in required columns
                        if self.df[required_cols].isnull().any().any():
                            self.logger.warning(f"Skipping {feature_name}: missing values in source columns")
                            continue
                            
                        self.df[feature_name] = operation()
                        self.logger.info(f"Created derived feature: {feature_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create {feature_name}: {str(e)}")
                        self.df[feature_name] = 0  # Default value
                else:
                    missing_cols = [col for col in required_cols if col not in self.df.columns]
                    self.logger.warning(f"Cannot create {feature_name}: missing {missing_cols}")
                    self.df[feature_name] = 0  # Default value

    def _encode_categorical_variables_production(self):
        """Encode categorical features using training encoders"""
        self.logger.info("Encoding categorical variables in production...")
        for col in self.cat_cols:
            if col in self.df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                
                # Convert to string and handle unseen categories
                self.df[col] = self.df[col].astype(str)
                
                # Create mapping for known classes
                encoding_map = {cls: idx for idx, cls in enumerate(le.classes_)}
                
                # Map values, assign -1 for unseen categories
                self.df[col] = self.df[col].map(encoding_map)
                
                # Handle unseen categories
                unseen_mask = self.df[col].isna()
                if unseen_mask.any():
                    self.logger.warning(f"Unseen categories in '{col}'. Encoding as -1.")
                    self.df.loc[unseen_mask, col] = -1
                
                self.df[col] = self.df[col].astype(int)

    def _feature_scaling_production(self):
        """Scale numerical features using training scaler"""
        self.logger.info("Scaling numerical features in production...")
        cols_to_scale = [col for col in self.num_cols if col in self.df.columns]
        
        if cols_to_scale and hasattr(self.scaler, 'mean_') and len(self.scaler.mean_) > 0:
            try:
                self.df[cols_to_scale] = self.scaler.transform(self.df[cols_to_scale])
            except ValueError as e:
                self.logger.error(f"Scaling failed: {str(e)}")
                # Fallback: use standard scaling with available parameters
                for col in cols_to_scale:
                    if col in self.scaler.mean_:
                        idx = list(self.scaler.mean_).index(col)
                        self.df[col] = (self.df[col] - self.scaler.mean_[idx]) / self.scaler.scale_[idx]

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure features match training feature order and presence"""
        # Remove target column for inference
        expected_features = [f for f in self.feature_names if f != self.target_col]
        
        # Add missing features with zeros
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0
                self.logger.warning(f"Missing feature '{feat}' - filled with 0")
        
        # Keep only expected features in the correct order
        df = df[expected_features]
        
        return df

    # Override parent methods that shouldn't be used in production
    def run_preprocessing_pipeline(self):
        raise NotImplementedError("Use preprocess() method for production inference")
    
    def handle_missing_values(self):
        raise NotImplementedError("Use _handle_missing_values_production() for production inference")
    
    def combine_cols(self):
        raise NotImplementedError("Use _combine_cols_production() for production inference")
    
    def encode_categorical_variables(self):
        raise NotImplementedError("Use _encode_categorical_variables_production() for production inference")
    
    def feature_scaling(self):
        raise NotImplementedError("Use _feature_scaling_production() for production inference")


def save_enhanced_preprocessing_artifacts(preprocessor_instance):
    """
    Enhanced artifact saving with complete reproduction capability
    """
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Store complete preprocessing state
    artifacts = {
        # Feature lists and configuration
        "potential_derived_source_cols":preprocessor_instance.features_to_drop,
        "feature_names": preprocessor_instance.df.columns.tolist(),
        "numerical_features": preprocessor_instance.num_cols,
        "categorical_features": preprocessor_instance.cat_cols,
        "target_column": preprocessor_instance.target_col,
        "drop_columns": preprocessor_instance.drop_col,
        
        # Transformation objects
        "label_encoders": {
            col: enc.classes_.tolist() 
            for col, enc in preprocessor_instance.label_encoders.items()
        },
        
        "scaler_params": {
            "mean": preprocessor_instance.scaler.mean_.tolist(),
            "scale": preprocessor_instance.scaler.scale_.tolist(),
            "var": preprocessor_instance.scaler.var_.tolist(),
            "n_samples_seen": int(preprocessor_instance.scaler.n_samples_seen_)
        },
        
        # Feature engineering parameters
        "feature_engineering_params": preprocessor_instance.feature_engineering_params,
        
        # Missing value handling
        "numerical_fill_values": preprocessor_instance.numerical_fill_values,
        "categorical_fill_values": preprocessor_instance.categorical_fill_values,
        
        # Target encoding
        "target_mapping": getattr(preprocessor_instance, 'target_mapping', {}),
        
        # Metadata
        "preprocessing_timestamp": datetime.utcnow().isoformat(),
        "n_samples": len(preprocessor_instance.df),
        "n_features": len(preprocessor_instance.df.columns),
        "feature_dtypes": {col: str(dtype) for col, dtype in preprocessor_instance.df.dtypes.items()}
    }
    
    # Convert all numpy types to native Python types
    artifacts = convert_numpy_types(artifacts)
    
    # Save artifacts as JSON
    artifacts_path = "src/data_pipeline/preprocessing_artifacts.json"
    os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
    
    with open(artifacts_path, "w") as f:
        json.dump(artifacts, f, indent=2, default=str)
    
    # Save pickle backup for complex objects
    pickle_path = "src/data_pipeline/preprocessing_artifacts.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "label_encoders": preprocessor_instance.label_encoders,
            "scaler": preprocessor_instance.scaler
        }, f)
    
    preprocessor_instance.logger.info(f"Enhanced artifacts saved to {artifacts_path}")
    return artifacts_path
