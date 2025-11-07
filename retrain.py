#!/usr/bin/env python3
"""
Retraining Pipeline for Churn Prediction Models
Orchestrates the full retraining workflow: data -> preprocessing -> training -> evaluation -> deployment
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_pipeline.pipeline_data import fetch_preprocessed
from src.models.train_nn import NeuralNetworkTrainer
from src.models.train_xgb import XGBoostTrainer, setup_logger as xgb_logger
from src.models.train_rf import evaluate_models as rf_evaluate
import torch

class ModelRetrainer:
    """
    Orchestrates retraining of all ML models with consistent pipeline
    """
    
    def __init__(self, config_path: str = "config/config_retrain.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.models_to_retrain = self.config.get("models_to_retrain", ["xgboost", "random_forest", "neural_net"])
        self.performance_threshold = self.config.get("performance_threshold", 0.7)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load retraining configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # Fallback to default config
            return {
                "models_to_retrain": ["xgboost", "random_forest", "neural_net"],
                "performance_threshold": 0.7,
                "enable_mlflow": True,
                "save_models": True
            }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for retraining pipeline"""
        log_dir = "logs/retraining"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"retrain_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality before retraining"""
        self.logger.info("Validating data quality...")
        
        checks = {
            "has_data": len(df) > 0,
            "has_features": len(df.columns) > 1,
            "no_nulls": not df.isnull().any().any(),
            "has_target": 'churn' in df.columns,
            "class_balance": df['churn'].value_counts().min() / len(df) > 0.1
        }
        
        for check_name, check_result in checks.items():
            if not check_result:
                self.logger.warning(f"Data quality check failed: {check_name}")
        
        return all(checks.values())
    def generate_test_data_simple(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Simple approach: use a stratified sample from the data"""
        from sklearn.model_selection import train_test_split
        
        # Create a proper test split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        test_data = {
            "samples": []
        }
        
        for idx in range(len(X_test)):
            sample = {
                "features": X_test.iloc[idx].to_dict(),
                "target": int(y_test.iloc[idx]),
                "sample_id": f"test_{idx}"
            }
            test_data["samples"].append(sample)
        
        # Save to file
        test_file_path = "test_input.json"
        with open(test_file_path, 'w') as f:
            import json
            json.dump(test_data, f, indent=2)
        
        return test_file_path
    
    def retrain_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Retrain XGBoost model"""
        self.logger.info("Starting XGBoost retraining...")
        
        try:
            # Load XGBoost config
            with open("config/config_train_xgb.yaml", "r") as f:
                xgb_config = yaml.safe_load(f)
            
            # Setup XGBoost logger
            xgb_logger = xgb_logger(
                xgb_config["logging"]["log_path"], 
                xgb_config["logging"]["log_level"]
            )
            
            trainer = XGBoostTrainer(config=xgb_config, logger=xgb_logger)
            model, metrics = trainer.train_and_tune_model(X, y)
            
            if model:
                trainer.save_model(model)
                self.logger.info("XGBoost retraining completed successfully")
                return {
                    "status": "success",
                    "metrics": metrics,
                    "model": "xgboost"
                }
            else:
                raise Exception("XGBoost training returned no model")
                
        except Exception as e:
            self.logger.error(f"XGBoost retraining failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "model": "xgboost"
            }
    
    def retrain_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Retrain Random Forest model"""
        self.logger.info("Starting Random Forest retraining...")
        
        try:
            # Load Random Forest config
            with open("config/config_train_rf.yaml", "r") as f:
                rf_config = yaml.safe_load(f)
            
            # Use the existing evaluate_models function
            _, best_model, best_metrics, _ = rf_evaluate(X, y, rf_config)
            
            if best_model:
                # Save model locally
                import joblib
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join("models", f"rf_model_{timestamp}.joblib")
                joblib.dump(best_model, model_path)
                
                self.logger.info("Random Forest retraining completed successfully")
                return {
                    "status": "success",
                    "metrics": best_metrics,
                    "model": "random_forest"
                }
            else:
                raise Exception("No suitable Random Forest model found")
                
        except Exception as e:
            self.logger.error(f"Random Forest retraining failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "model": "random_forest"
            }
    
    def retrain_neural_network(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Retrain Neural Network model"""
        self.logger.info("Starting Neural Network retraining...")
        
        try:
            # Load NN config
            with open("config/config_train_nn.yaml", "r") as f:
                nn_config = yaml.safe_load(f)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trainer = NeuralNetworkTrainer(X, y, nn_config, device)
            
            # Train and tune
            trainer.train_and_tune()
            
            # Save model
            model_path = trainer.save_model()
            
            self.logger.info("Neural Network retraining completed successfully")
            return {
                "status": "success",
                "metrics": {"f1": 0.0},  
                "model": "neural_net",
                "model_path": model_path
            }
            
        except Exception as e:
            self.logger.error(f"Neural Network retraining failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "model": "neural_net"
            }
    
    def evaluate_model_performance(self, results: List[Dict]) -> Dict:
        """Evaluate if retrained models meet performance thresholds"""
        self.logger.info("Evaluating model performance...")
        
        evaluation = {
            "total_models": len(results),
            "successful_models": 0,
            "failed_models": 0,
            "models_meeting_threshold": 0,
            "best_model": None,
            "best_score": 0
        }
        
        for result in results:
            if result["status"] == "success":
                evaluation["successful_models"] += 1
                
                # Check if metrics meet threshold (simplified)
                metrics = result.get("metrics", {})
                f1_score = metrics.get("f1_score", 0)
                
                if f1_score >= self.performance_threshold:
                    evaluation["models_meeting_threshold"] += 1
                    
                    # Track best model
                    if f1_score > evaluation["best_score"]:
                        evaluation["best_score"] = f1_score
                        evaluation["best_model"] = result["model"]
            else:
                evaluation["failed_models"] += 1
        
        return evaluation
    
    def run_retraining_pipeline(self) -> Dict:
        """Execute full retraining pipeline"""
        self.logger.info("Starting full model retraining pipeline...")
        
        start_time = datetime.now()
        results = []
        
        try:
            # Step 1: Fetch and validate data
            self.logger.info("Fetching preprocessed data...")
            
            if os.path.exists("data/processed/processed_data.csv"):
                df_processed = pd.read_csv("data/processed/processed_data.csv")
            else: df_processed = fetch_preprocessed()
            
            if not self.validate_data_quality(df_processed):
                self.logger.warning("Data quality issues detected, but continuing...")
            
            # Prepare features and target
            target_col = 'churn'
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
            
            self.logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            self.generate_test_data_simple(X, y)

            # Step 2: Retrain models
            self.logger.info("Retraining models...")
            
            if "xgboost" in self.models_to_retrain:
                results.append(self.retrain_xgboost(X, y))
            
            if "random_forest" in self.models_to_retrain:
                results.append(self.retrain_random_forest(X, y))
            
            if "neural_net" in self.models_to_retrain:
                results.append(self.retrain_neural_network(X, y))
            
            # Step 3: Evaluate results
            self.logger.info("Evaluating retraining results...")
            evaluation = self.evaluate_model_performance(results)
            
            # Step 4: Generate report
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            final_report = {
                "retraining_id": start_time.strftime("%Y%m%d_%H%M%S"),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "models_retrained": self.models_to_retrain,
                "results": results,
                "evaluation": evaluation,
                "overall_status": "success" if evaluation["successful_models"] > 0 else "failed"
            }
            
            self.logger.info(f"Retraining pipeline completed in {duration:.2f} seconds")
            self.logger.info(f"Results: {evaluation['successful_models']}/{evaluation['total_models']} models successful")
            
            # Save report
            self._save_retraining_report(final_report)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Retraining pipeline failed: {str(e)}")
            return {
                "retraining_id": start_time.strftime("%Y%m%d_%H%M%S"),
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "error": str(e),
                "overall_status": "failed"
            }
    
    def _save_retraining_report(self, report: Dict):
        """Save retraining report to file"""
        reports_dir = "artifacts/retraining_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        report_file = os.path.join(reports_dir, f"retraining_report_{report['retraining_id']}.json")
        
        try:
            import json
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Retraining report saved to {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save retraining report: {str(e)}")

def main():
    """Main entry point for retraining script"""
    parser = argparse.ArgumentParser(description="Retrain churn prediction models")
    parser.add_argument("--config", type=str, default="config/config_retrain.yaml", 
                       help="Path to retraining configuration file")
    parser.add_argument("--models", type=str, nargs="+", 
                       choices=["xgboost", "random_forest", "neural_net", "all"],
                       help="Specific models to retrain")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Performance threshold for model acceptance")
    
    args = parser.parse_args()
    
    # Initialize retrainer
    retrainer = ModelRetrainer(args.config)
    
    # Override models if specified
    if args.models:
        if "all" in args.models:
            retrainer.models_to_retrain = ["xgboost", "random_forest", "neural_net"]
        else:
            retrainer.models_to_retrain = args.models
    
    # Override threshold if specified
    if args.threshold:
        retrainer.performance_threshold = args.threshold
    
    # Run retraining pipeline
    report = retrainer.run_retraining_pipeline()
    
    # Exit with appropriate code
    if report["overall_status"] == "success":
        print("Retraining completed successfully!")
        sys.exit(0)
    else:
        print("Retraining failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()