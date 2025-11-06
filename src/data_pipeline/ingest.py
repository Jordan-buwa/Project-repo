import os
import yaml
import pandas as pd
import hashlib
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

def setup_logger(log_path: str, log_level: str = "INFO"):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path+f"ingest_{timestamp}.log",
        filemode="a", 
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)

def compute_file_hash(path: str) -> str:
    """Generate SHA256 hash for a file (used for metadata tracking)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def save_metadata(meta_path: str, record: dict):
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = []
    else:
        metadata = []
    metadata.append(record)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

# Data Ingestion Class

class DataIngestion:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.source_type = self.config["data_source"]["type"]
        self.logger = setup_logger(
            self.config["logging"]["log_path"],
            self.config["logging"]["log_level"]
        )

        self.output_dir = self.config["storage"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Ingest data based on source type (csv, database)."""
        self.logger.info(f"Starting data ingestion from source: {self.source_type}")

        if self.source_type == "csv":
            df = self._load_csv()
        elif self.source_type == "database":
            df = self._load_database()
        else:
            raise ValueError(f"Unsupported data source type: {self.source_type}")

        # Save raw snapshot
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"{self.config['file_name']}_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        self.logger.info(f"Data snapshot saved: {output_file}")

        # Track metadata
        if self.config["metadata"]["save_metadata"]:
            self._log_metadata(output_file, df)

        # DVC tracking
        if self.config["storage"]["dvc_track"]:
            os.system(f"dvc add {output_file}")

        return df

    # Source-specific Loaders

    def _load_csv(self) -> pd.DataFrame:
        cfg = self.config["csv"]
        backup_dir = self.config["storage"]["backup_dir"]
        os.makedirs(backup_dir, exist_ok=True)

        # Check for training & testing paths
        train_path = cfg.get("train_path")
        test_path = cfg.get("test_path")
        single_path = cfg.get("path")

        if train_path and test_path:
            self.logger.info(f"Loading training data from: {train_path}")
            self.logger.info(f"Loading testing data from: {test_path}")

            try:
                df_train = pd.read_csv(
                    train_path,
                    delimiter=cfg.get("delimiter", ","),
                    encoding=cfg.get("encoding", "utf-8")
                )
                df_test = pd.read_csv(
                    test_path,
                    delimiter=cfg.get("delimiter", ","),
                    encoding=cfg.get("encoding", "utf-8")
                )

                self.logger.info(
                    f"Loaded {len(df_train)} training records and {len(df_test)} testing records."
                )

                # Combine or return as dict (you can decide)
                df = pd.concat(
                    [df_train.assign(dataset="train"), df_test.assign(dataset="test")],
                    ignore_index=True
                )
                self.logger.info("Training and testing datasets concatenated successfully.")
                df.to_csv(backup_dir + "ingested.csv", index=False, encoding='utf-8', sep=',', header=True)
                return df

            except Exception as e:
                self.logger.error(f"Error loading train/test CSVs: {e}")
                raise

        elif single_path:
            self.logger.info(f"Loading data from: {single_path}")
            try:
                df = pd.read_csv(
                    single_path,
                    delimiter=cfg.get("delimiter", ","),
                    encoding=cfg.get("encoding", "utf-8")
                )
                self.logger.info(f"Loaded {len(df)} records from {single_path}")
                print(f"Loaded {len(df)} records from {single_path}")
                df.to_csv(backup_dir + "ingested.csv", index=False, encoding='utf-8', sep=',', header=True)
                return df
            except Exception as e:
                self.logger.error(f"CSV ingestion failed for {single_path}: {e}")
                raise

        else:
            msg = "No valid CSV path(s) found in config file. Expected 'path' or ('train_path' and 'test_path')."
            self.logger.error(msg)
            raise ValueError(msg)

    def _load_database(self) -> pd.DataFrame:
        """Load data from a PostgreSQL database using SQLAlchemy."""

        db_cfg = self.config["database"]["db_config"]
        query = self.config["database"]["query"]
        fetch_size = self.config["database"].get("fetch_size", 10000)

        # Read password securely from environment variable
        password = os.getenv("POSTGRES_PASSWORD")

        # Build SQLAlchemy connection string
        conn_str = (
            f"{db_cfg['dialect']}+{db_cfg['driver']}://"
            f"{db_cfg['user']}:{password}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
            f"?sslmode={db_cfg.get('sslmode', 'require')}"
        )

        self.logger.info(f"Connecting to database host: {db_cfg['host']}:{db_cfg['port']}")
        # self.logger.debug(f"Connection string: {conn_str.replace(password, '***')}")  # hide password

        try:
            engine = create_engine(conn_str)

            # Stream results if dataset is large
            with engine.connect() as conn:
                self.logger.info(f"Executing query: {query[:100]}...")  # log partial query
                df_iter = pd.read_sql_query(query, conn, chunksize=fetch_size)

                # Combine chunks if necessary
                if isinstance(df_iter, pd.DataFrame):
                    df = df_iter
                else:
                    df = pd.concat(df_iter, ignore_index=True)

            self.logger.info(f"Fetched {len(df)} records from database.")
            return df

        except Exception as e:
            self.logger.error(f"Database ingestion failed: {e}")
            if self.config["logging"].get("save_failed_batches", False):
                failed_dir = self.config["logging"].get("failed_dir", "data/ingestion/failed_batches/")
                os.makedirs(failed_dir, exist_ok=True)
                failed_file = os.path.join(failed_dir, "failed_query.txt")
                with open(failed_file, "w") as f:
                    f.write(query)
                self.logger.warning(f"Saved failed query to {failed_file}")
            raise

    # Metadata Logger

    def _log_metadata(self, file_path: str, df: pd.DataFrame):
        meta_cfg = self.config["metadata"]
        meta_path = meta_cfg["path"]
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_type": self.source_type,
            "records": len(df),
            "file_path": file_path,
            "file_hash": compute_file_hash(file_path)
        }
        save_metadata(meta_path, record)
        self.logger.info(f"Metadata recorded: {record}")

# Entrypoint

if __name__ == "__main__":
    CONFIG_PATH = "config/config_ingest.yaml"
    ingestion = DataIngestion(CONFIG_PATH)
    df = ingestion.load_data()
    print(f"Ingested {len(df)} records from {ingestion.source_type}")
