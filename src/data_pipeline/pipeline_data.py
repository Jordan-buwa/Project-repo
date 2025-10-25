from src.data_pipeline.ingest import DataIngestion
from src.data_pipeline.preprocess import DataPreprocessor


def fetch_preprocessed():
    ingestion = DataIngestion("config/config_ingest.yaml")
    df_raw = ingestion.load_data()
    df_processed = DataPreprocessor(
        "config/config_process.yaml", data_raw=df_raw)
    return df_processed
