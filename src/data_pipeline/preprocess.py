#  Data Preprocessor Class
import pandas as pd


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
        os.makedirs("data/processed", exist_ok=True)
        self.logger = setup_logger("src/data/logs/preprocessing.log")

    # Feature Engineering

    def combine_cols(self):
        """Create derived features"""
        self.df["engagement_index"] = (
            self.df["outcalls"] + self.df["incalls"]) / (self.df["months"] + 1)
        self.df["model_change_rate"] = self.df["models"] / \
            (self.df["months"] + 1)
        self.df["overage_ratio"] = self.df["overage"] / \
            (self.df["revenue"] + 1)
        self.logger.info("Derived features added successfully.")
        self.num_cols += self.config["combined_features"]      

    def remove_unnecessary_columns(self):
        """Drop unneeded columns"""
        for col in self.drop_col:
            if col in self.df.columns:
                self.df.drop(columns=col, inplace=True)
                self.logger.info(f"Removed column: {col}")

    def handle_missing_values(self):
        """Fill missing values for numerical and categorical"""
        self.logger.info("Handling missing values...")
        for col in self.num_cols:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].median(), inplace=True)

        for col in self.cat_cols:
            if self.df[col].isnull().any():
                fill_value = self.df[col].mode(
                )[0] if not self.df[col].mode().empty else "Unknown"
                self.df[col].fillna(fill_value, inplace=True)

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

        with open("data/processed/preprocessing_artifacts.json", "w") as f:
            json.dump(artifacts, f, indent=2)
        self.logger.info("Processed data saved locally.")

        # PostgreSQL snapshot (optional)
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
        self.combine_cols()
        self.remove_unnecessary_columns()
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.encode_target_variable()
        self.feature_scaling()
        self.save_preprocessed_data()

        if self.df.isnull().any().any():
            self.logger.error("NaN values detected after preprocessing!")
            raise ValueError(
                "Data contains NaNs after preprocessing pipeline.")

        self.logger.info("Preprocessing pipeline completed successfully.")
        return self.df
