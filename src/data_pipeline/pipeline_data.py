import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_pipeline.ingest import DataIngestion
from src.data_pipeline.preprocess import DataPreprocessor
from src.data_pipeline.validate_after_preprocess import validate_dataframe


def fetch_preprocessed():
    ingestion = DataIngestion("config/config_ingest.yaml")
    df_raw = ingestion.load_data()
    df_processed = DataPreprocessor(
        "config/config_process.yaml", data_raw=df_raw).run_preprocessing_pipeline()
    df = validate_dataframe(df_processed, "config/config_process.yaml")
    return df

