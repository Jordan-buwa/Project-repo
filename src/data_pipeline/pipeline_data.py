import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_pipeline.ingest import DataIngestion
from src.data_pipeline.preprocess import DataPreprocessor, save_enhanced_preprocessing_artifacts
from src.data_pipeline.validate_after_preprocess import validate_dataframe


def fetch_preprocessed():
    ingestion = DataIngestion("config/config_ingest.yaml")
    df_raw = ingestion.load_data()
    processor = DataPreprocessor(
        "config/config_process.yaml", data_raw=df_raw)
    df_processed = processor.run_preprocessing_pipeline()
    save_enhanced_preprocessing_artifacts(processor) 

    df = validate_dataframe(df_processed, "config/config_process.yaml", processor)
    df.to_csv("data/processed/processed_data.csv", sep = ",", encoding="utf8", index=False)
    return df

