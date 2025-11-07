#!/usr/bin/env python3
"""
Main entry point for the Data Pipeline Docker container.
Clean interface for all data pipeline operations.
"""

import argparse
import sys
import logging
import pandas as pd
from pathlib import Path
import time

def setup_logging():
    """Setup logging configuration that works in both Docker and local environments"""
    # Determining log directory based on environment
    if Path("/app").exists():  # Docker environment
        log_dir = Path("/app/logs")
    else:  # Local development
        log_dir = Path("logs")
    
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "pipeline_main.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_data_simulation(args):
    """Run data simulation with error handling"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data simulation...")
    
    try:
        from src.data_pipeline.data_simulation import RealisticDataSimulator, DriftedDataSimulator, main as sim_main
        
        # Choosing simulator type
        if args.simulator_type == "drifted":
            simulator = DriftedDataSimulator(random_state=args.random_state)
            # Configuring drift with moderate settings
            simulator.configure_drift(
                target_drift_strength=0.3,
                feature_drift_strength=0.4,
                categorical_drift_strength=0.5
            )
        else:
            simulator = RealisticDataSimulator(random_state=args.random_state)
        
        # Fitting the simulator
        logger.info(f"Fitting simulator with data from: {args.data_path}")
        simulator.fit(
            data_path=args.data_path,
            target=args.target,
            sample_size=args.sample_size
        )
        
        # Generating synthetic data
        logger.info(f"Generating {args.n_samples} synthetic samples...")
        synthetic_data = simulator.simulate(n_samples=args.n_samples)
        
        # Saving or return data
        if args.output_path:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            synthetic_data.to_csv(output_path, index=False)
            logger.info(f"✓ Saved synthetic data to: {output_path}")
        
        logger.info(f"✓ Successfully generated {len(synthetic_data)} samples")
        logger.info(f"✓ Data shape: {synthetic_data.shape}")
        logger.info(f"✓ Columns: {list(synthetic_data.columns)}")
        
        # Showing sample statistics
        if args.target in synthetic_data.columns:
            target_dist = synthetic_data[args.target].value_counts()
            logger.info(f"✓ Target distribution:\n{target_dist}")
        
        return synthetic_data
        
    except Exception as e:
        logger.error(f"Data simulation failed: {e}")
        raise

def run_ingestion(args):
    """Run data ingestion pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data ingestion...")
    
    try:
        # Importing and run ingestion - ingest.py has DataIngestion class
        from src.data_pipeline.ingest import DataIngestion
        
        logger.info("Running ingestion pipeline...")
        ingestion = DataIngestion("config/config_ingest.yaml")
        df = ingestion.load_data()
        logger.info(f"✓ Data ingestion completed successfully. Loaded {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

def run_preprocessing(args):
    """Run data preprocessing pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing...")
    
    try:
        # Importing and running preprocessing - preprocess.py has DataPreprocessor class
        from src.data_pipeline.preprocess import DataPreprocessor, save_enhanced_preprocessing_artifacts
        
        logger.info("Running preprocessing pipeline...")
        
        # Loading data from ingestion output
        try:
            df = pd.read_csv("data/backup/ingested.csv")
            logger.info(f"Loaded {len(df)} records from ingested data")
        except FileNotFoundError:
            logger.warning("No ingested data found. Running ingestion first...")
            df = run_ingestion(args)
        
        # Running preprocessing
        preprocessor = DataPreprocessor("config/config_process.yaml", df)
        processed_df = preprocessor.run_preprocessing_pipeline()
        
        # Saving artifacts for production
        save_enhanced_preprocessing_artifacts(preprocessor)
        logger.info("Preprocessing artifacts saved")
        
        logger.info(f"Data preprocessing completed successfully. Processed {len(processed_df)} records")
        return processed_df
        
    except Exception as e:
        logger.error(f" Data preprocessing failed: {e}")
        raise

def run_validation(args):
    """Run data validation after preprocessing"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data validation...")
    
    try:
        # Importing validation functions
        from src.data_pipeline.validate_after_preprocess import validate_dataframe
        from src.data_pipeline.preprocess import DataPreprocessor
        
        logger.info("Running data validation...")
        
        # Loading processed data
        try:
            df = pd.read_csv("data/processed/processed_data.csv")
            logger.info(f"Loaded {len(df)} records for validation")
        except FileNotFoundError:
            logger.warning("No processed data found. Running preprocessing first...")
            df = run_preprocessing(args)
        
        # Creating preprocessor instance for validation
        preprocessor = DataPreprocessor("config/config_process.yaml", df)
        
        # Running validation
        validated_df = validate_dataframe(df, "config/config_process.yaml", preprocessor)
        
        logger.info(f"✓ Data validation completed successfully. Validated {len(validated_df)} records")
        return validated_df
        
    except Exception as e:
        logger.error(f" Data validation failed: {e}")
        raise

def run_full_pipeline(args):
    """Run the complete data pipeline: ingest → preprocess → validate"""
    logger = logging.getLogger(__name__)
    logger.info("Starting FULL data pipeline...")
    
    try:
        start_time = time.time()
        
        # Running ingestion
        logger.info("=== STEP 1: Data Ingestion ===")
        df_ingested = run_ingestion(args)
        
        # Running preprocessing
        logger.info("=== STEP 2: Data Preprocessing ===")
        df_processed = run_preprocessing(args)
        
        # Running validation
        logger.info("=== STEP 3: Data Validation ===")
        df_validated = run_validation(args)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ FULL data pipeline completed successfully in {elapsed_time:.2f} seconds!")
        logger.info(f"✓ Final dataset: {df_validated.shape[0]} rows, {df_validated.shape[1]} columns")
        
        return df_validated
        
    except Exception as e:
        logger.error(f"Full pipeline failed: {e}")
        raise

def test_module_imports():
    """Test if all required modules can be imported"""
    logger = logging.getLogger(__name__)
    logger.info("Testing module imports...")
    
    modules_to_test = [
        "src.data_pipeline.data_simulation",
        "src.data_pipeline.ingest", 
        "src.data_pipeline.preprocess",
        "src.data_pipeline.validate_after_preprocess"
    ]
    
    all_imports_ok = True
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"✓ {module_name}")
        except ImportError as e:
            logger.error(f"✗ {module_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def main():
    """Main entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description="Data Pipeline Main Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py simulate --n-samples 1000
  python main.py pipeline
  python main.py ingest
  python main.py preprocess
  python main.py validate
  python main.py test-imports
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Simulation command
    sim_parser = subparsers.add_parser("simulate", help="Generate synthetic data")
    sim_parser.add_argument("--simulator-type", choices=["realistic", "drifted"], 
                          default="realistic", help="Type of simulator")
    sim_parser.add_argument("--data-path", default="data/raw/telco_churn.csv", 
                          help="Path to training data")
    sim_parser.add_argument("--target", default="churn", help="Target column name")
    sim_parser.add_argument("--sample-size", type=int, default=500, 
                          help="Sample size for fitting simulator")
    sim_parser.add_argument("--n-samples", type=int, default=1000, 
                          help="Number of synthetic samples to generate")
    sim_parser.add_argument("--output-path", help="Path to save synthetic data")
    sim_parser.add_argument("--random-state", type=int, default=42, 
                          help="Random seed")
    
    # Individual pipeline commands
    subparsers.add_parser("ingest", help="Run data ingestion pipeline")
    subparsers.add_parser("preprocess", help="Run data preprocessing pipeline")  
    subparsers.add_parser("validate", help="Run data validation")
    
    # Full pipeline command
    subparsers.add_parser("pipeline", help="Run complete data pipeline (ingest → preprocess → validate)")
    
    # Test command
    subparsers.add_parser("test-imports", help="Test if all modules can be imported")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        logger.info(f"Executing command: {args.command}")
        
        if args.command == "simulate":
            run_data_simulation(args)
        elif args.command == "ingest":
            run_ingestion(args)
        elif args.command == "preprocess":
            run_preprocessing(args)
        elif args.command == "validate":
            run_validation(args)
        elif args.command == "pipeline":
            run_full_pipeline(args)
        elif args.command == "test-imports":
            test_module_imports()
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            
        logger.info(f"✓ Command '{args.command}' completed successfully")
        
    except Exception as e:
        logger.error(f"Command '{args.command}' failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()