import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

class SetupValidator:
    """Validate API setup and dependencies"""
    
    def __init__(self):
        self.repo_root = Path(__file__).resolve().parent.parent.parent.parent
        self.errors = []
        self.warnings = []
    
    def validate_imports(self) -> bool:
        """Validate that all critical imports work"""
        logger.info("Validating imports...")
        
        import_checks = [
            ("model_types", "from .model_types import ModelType, validate_model_type"),
            ("ml_models", "from ..ml_models import get_model_path, load_all_models"),
            ("config", "from .config import APIConfig"),
            ("customer_data", "from .customer_data import CustomerData"),
            ("database", "from .database import get_db_connection"),
            ("error_handlers", "from .error_handlers import APIError, handle_model_error"),
        ]
        
        success = True
        for module_name, import_stmt in import_checks:
            try:
                exec(import_stmt)
                logger.info(f"SUCCESS: {module_name} imported successfully")
            except ImportError as e:
                logger.error(f"FAILED: Failed to import {module_name}: {e}")
                self.errors.append(f"Import failed for {module_name}: {str(e)}")
                success = False
        
        return success
    
    def validate_paths(self) -> bool:
        """Validate that all required paths exist"""
        logger.info("Validating paths...")
        
        required_dirs = [
            self.repo_root / "models",
            self.repo_root / "src" / "api" / "cache",
            self.repo_root / "src" / "api" / "logs",
            self.repo_root / "config",
        ]
        
        success = True
        for directory in required_dirs:
            if directory.exists():
                logger.info(f"SUCCESS: Directory exists: {directory}")
            else:
                logger.warning(f"WARNING: Directory missing: {directory}")
                self.warnings.append(f"Directory missing: {directory}")
                # Don't fail for missing directories, just warn
        
        # Critical files that must exist
        critical_files = [
            self.repo_root / "src" / "data_pipeline" / "preprocessing_artifacts.json",
            self.repo_root / "test_input.json",
        ]
        
        for file_path in critical_files:
            if file_path.exists():
                logger.info(f"SUCCESS: File exists: {file_path}")
            else:
                logger.error(f"FAILED: Critical file missing: {file_path}")
                self.errors.append(f"Critical file missing: {file_path}")
                success = False
        
        return success
    
    def validate_model_types(self) -> bool:
        """Validate model type consistency"""
        logger.info("Validating model types...")
        
        try:
            from .models_types import ModelType, normalize_model_type, validate_model_type
            
            # Test normalization
            test_cases = [
                ("xgboost", "xgboost"),
                ("random-forest", "random_forest"),
                ("random_forest", "random_forest"), 
                ("neural-net", "neural_net"),
            ]
            
            success = True
            for input_type, expected in test_cases:
                result = normalize_model_type(input_type)
                if result != expected:
                    logger.error(f"FAILED: Normalization failed: {input_type} -> {result} (expected {expected})")
                    self.errors.append(f"Model type normalization failed for {input_type}")
                    success = False
                else:
                    logger.info(f"SUCCESS: Normalization correct: {input_type} -> {result}")
            
            # Test validation
            valid_types = ["xgboost", "random_forest", "neural_net", "random-forest", "neural-net"]
            invalid_types = ["invalid_model", "unknown", "xg_boost"]
            
            for model_type in valid_types:
                if validate_model_type(model_type):
                    logger.info(f"SUCCESS: Validation correct for: {model_type}")
                else:
                    logger.error(f"FAILED: Validation failed for valid type: {model_type}")
                    success = False
            
            for model_type in invalid_types:
                if not validate_model_type(model_type):
                    logger.info(f"SUCCESS: Correctly rejected invalid type: {model_type}")
                else:
                    logger.error(f"FAILED: Wrongly accepted invalid type: {model_type}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"FAILED: Model type validation failed: {e}")
            self.errors.append(f"Model type validation failed: {str(e)}")
            return False
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        logger.info("Validating configuration...")
        
        try:
            from .config import APIConfig
            
            config = APIConfig()
            logger.info("SUCCESS: Configuration loaded successfully")
            
            # Test config properties
            test_properties = [
                "model_dir",
                "cache_dir", 
                "logs_dir",
                "preprocessing_artifacts_path",
                "test_data_path"
            ]
            
            for prop in test_properties:
                try:
                    value = getattr(config, prop)
                    logger.info(f"SUCCESS: Config property {prop}: {value}")
                except Exception as e:
                    logger.error(f"FAILED: Config property {prop} failed: {e}")
                    self.errors.append(f"Config property {prop} failed: {str(e)}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"FAILED: Configuration validation failed: {e}")
            self.errors.append(f"Configuration validation failed: {str(e)}")
            return False
    
    def validate_database(self) -> bool:
        """Validate database connection"""
        logger.info("Validating database connection...")
        
        try:
            from .database import get_db_connection, initialize_connection_pool
            
            # Test connection
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                
            if result and result[0] == 1:
                logger.info("SUCCESS: Database connection test passed")
                return True
            else:
                logger.error("FAILED: Database connection test failed")
                self.errors.append("Database connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"FAILED: Database connection failed: {e}")
            self.errors.append(f"Database connection failed: {str(e)}")
            return False
    
    def validate_ml_models(self) -> bool:
        """Validate ML models functionality"""
        logger.info("Validating ML models functionality...")
        
        try:
            from ..ml_models import get_model_path, check_models_availability
            
            # Check model availability
            availability = check_models_availability()
            logger.info(f"Model availability: {availability}")
            
            # Test get_model_path for each model type
            model_types = ["xgboost", "random_forest", "neural_net"]
            
            for model_type in model_types:
                try:
                    path = get_model_path(model_type)
                    logger.info(f"SUCCESS: Model path found for {model_type}: {path}")
                except Exception as e:
                    logger.warning(f"WARNING: Could not find model path for {model_type}: {e}")
                    self.warnings.append(f"Model path not found for {model_type}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"FAILED: ML models validation failed: {e}")
            self.errors.append(f"ML models validation failed: {str(e)}")
            return False
    
    def run_all_checks(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validation checks"""
        logger.info("Starting comprehensive API setup validation...")
        
        checks = [
            self.validate_imports(),
            self.validate_paths(),
            self.validate_model_types(), 
            self.validate_config(),
            self.validate_database(),
            self.validate_ml_models(),
        ]
        
        all_passed = all(checks)
        
        if all_passed:
            logger.info("ALL CHECKS PASSED: API should start successfully.")
        else:
            logger.error("SOME CHECKS FAILED: Please fix the errors above.")
        
        return all_passed, self.errors, self.warnings

# Convenience function
def validate_api_setup() -> Tuple[bool, List[str], List[str]]:
    """Convenience function to run all validations"""
    validator = SetupValidator()
    return validator.run_all_checks()