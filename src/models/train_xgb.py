import subprocess
import warnings
from src.data_pipeline.pipeline_data import fetch_preprocessed
from collections import defaultdict
import os
import sys
import yaml
import joblib
import logging
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.combine import SMOTETomek
import mlflow
import traceback
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
import json
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()
[warnings.filterwarnings("ignore", category=c)
 for c in (UserWarning, FutureWarning)]
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


# Logger Setup
def setup_logger(log_path: str, log_level: str = "INFO"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_path = log_path.replace(".log", f"_{timestamp}.log")
    logging.basicConfig(
        filename=full_log_path,
        filemode="a",
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Also add console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.addHandler(console_handler)

    return logger


class XGBoostTrainer:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.logger.info("XGBoost trainer initialized")

    @staticmethod
    def find_best_threshold(y_true, y_probs):
        best_thresh, best_f1 = 0.5, 0
        for t in [i * 0.01 for i in range(1, 100)]:
            preds = (y_probs >= t).astype(int)
            score = f1_score(y_true, preds)
            if score > best_f1:
                best_f1 = score
                best_thresh = t
        return best_thresh, best_f1

    def _validate_data(self, X, y):
        """Validate input data before training"""
        self.logger.info("Validating input data...")

        # Check for NaN values
        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            self.logger.error(f"NaN values found in features: {nan_cols}")
            raise ValueError(
                f"Data contains NaN values in columns: {nan_cols}")

        if y.isnull().any():
            self.logger.error("NaN values found in target variable")
            raise ValueError("Target variable contains NaN values")

        # Check data types
        if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            non_numeric_cols = X.columns[~X.dtypes.apply(
                lambda x: np.issubdtype(x, np.number))].tolist()
            self.logger.error(f"Non-numeric columns found: {non_numeric_cols}")
            raise ValueError(f"Non-numeric columns: {non_numeric_cols}")

        # Check target distribution
        target_distribution = y.value_counts()
        self.logger.info(f"Target distribution:\n{target_distribution}")

        if len(target_distribution) < 2:
            self.logger.error("Target variable has only one class")
            raise ValueError("Target variable must have at least two classes")

    def train_and_tune_model(self, X, y):
        self.logger.info("Starting model training with Stratified K-Fold...")

        skf = StratifiedKFold(
            n_splits=self.config["cv_folds"],
            shuffle=True,
            random_state=self.config["random_state"]
        )

        fold_metrics = []
        param_grid = self.config["xgboost_params"]

        # Fetch DVC hash for tracking
        try:
            dvc_hash = subprocess.getoutput(
                "dvc hash data/processed/preprocessed.csv")
        except Exception:
            dvc_hash = "N/A"

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
        mlflow.set_tracking_uri(mlflow_uri)

        experiment_name = "XGBoost_Churn_Experiment"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            self.logger.warning(f"Could not set up MLflow experiment: {e}")

        y_true_global = []
        y_pred_global = []

        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(self.config)
            mlflow.set_tag("dvc_data_hash", dvc_hash)

            #  K-Fold Training
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                self.logger.info(f"Starting fold {fold}...")
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Resampling inside fold
                if self.config.get("apply_smotetomek", True):
                    smt = SMOTETomek(random_state=self.config["random_state"])
                    X_train, y_train = smt.fit_resample(X_train, y_train)
                    self.logger.info(
                        f"Fold {fold}: Training size after SMOTETomek: {X_train.shape[0]}")

                try:
                    xgb = XGBClassifier(
                        objective='binary:logistic',
                        eval_metric='logloss',
                        random_state=self.config["random_state"],
                        n_jobs=-1  # Use all available cores
                    )

                    tuner = RandomizedSearchCV(
                        estimator=xgb,
                        param_distributions=param_grid,
                        scoring='f1',
                        n_iter=self.config.get("n_iter", 10),
                        cv=3,
                        n_jobs=-1,
                        random_state=self.config["random_state"],
                        verbose=1  # Add progress output
                    )

                    self.logger.info(
                        f"Starting hyperparameter tuning for fold {fold}...")
                    tuner.fit(X_train, y_train)

                except Exception as e:
                    self.logger.error(f"Fold {fold} failed: {e}")
                    self.logger.error(traceback.format_exc())

                    continue  # Skip this fold and continue with others
                best_model = tuner.best_estimator_

                y_probs = best_model.predict_proba(X_val)[:, 1]
                best_threshold, best_f1 = self.find_best_threshold(
                    y_val, y_probs)
                y_pred = (y_probs >= best_threshold).astype(int)

                # Collect predictions for global F1
                y_true_global.extend(y_val)
                y_pred_global.extend(y_pred)

                acc = accuracy_score(y_val, y_pred)
                roc = roc_auc_score(y_val, y_probs)
                self.logger.info(
                    f"Fold {fold}: Accuracy={acc:.4f}, F1={best_f1:.4f}, ROC-AUC={roc:.4f}")

                mlflow.log_metrics({
                    f"fold_{fold}_accuracy": acc,
                    f"fold_{fold}_f1": best_f1,
                    f"fold_{fold}_roc_auc": roc,
                })
                #  MLflow Model Logging
                fold_input_example = X_val.head(5)
                mlflow.xgboost.log_model(
                    best_model, name=f"xgboost_model_fold_{fold}", input_example=fold_input_example, registered_model_name="Churn_XGB_Model")

                fold_metrics.append({
                    "fold": fold,
                    "best_params": tuner.best_params_,
                    "accuracy": acc,
                    "f1_score": best_f1,
                    "roc_auc": roc,
                    "threshold": best_threshold,
                    "y_val_list": y_val.tolist(),
                    "y_pred_list": y_pred.tolist()
                })

            # Compute Global F1
            global_f1 = f1_score(
                y_true_global, y_pred_global, average="binary")
            self.logger.info(f"Global F1 across all folds: {global_f1:.4f}")
            mlflow.log_metric("global_f1", global_f1)

            # Select Best Params Using Global F1
            param_preds = defaultdict(lambda: {"y_true": [], "y_pred": []})

            for m in fold_metrics:
                key = tuple(sorted(m["best_params"].items()))
                param_preds[key]["y_true"].extend(m["y_val_list"])
                param_preds[key]["y_pred"].extend(m["y_pred_list"])

            global_f1_per_param = {
                k: f1_score(v["y_true"], v["y_pred"], average="binary")
                for k, v in param_preds.items()
            }

            best_params_tuple = max(
                global_f1_per_param, key=global_f1_per_param.get)
            best_params = dict(best_params_tuple)
            best_global_f1 = global_f1_per_param[best_params_tuple]

            self.logger.info(f"Best parameters (global F1): {best_params}")
            self.logger.info(
                f"Best global F1 across folds: {best_global_f1:.4f}")
            mlflow.log_metric("best_global_f1", best_global_f1)

            #  Train Final Model
            if self.config.get("apply_smotetomek", True):
                try:
                    smt = SMOTETomek(random_state=self.config["random_state"])
                    X_train, y_train = smt.fit_resample(X_train, y_train)
                    self.logger.info(
                        f"Fold {fold}: Training size after SMOTETomek: {X_train.shape[0]}")
                except Exception as e:
                    self.logger.error(f"SMOTETomek failed in fold {fold}: {e}")
                    # Continue without resampling
                    self.logger.info(
                        "Continuing without SMOTETomek resampling")

            final_model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=self.config["random_state"],
                **best_params
            )
            final_model.fit(X, y)

            y_probs_full = final_model.predict_proba(X)[:, 1]
            best_threshold, best_f1 = self.find_best_threshold(y, y_probs_full)
            final_model.threshold = best_threshold
            y_pred_full = (y_probs_full >= best_threshold).astype(int)

            acc = accuracy_score(y, y_pred_full)
            roc = roc_auc_score(y, y_probs_full)
            self.logger.info(
                f"Final model: Accuracy={acc:.4f}, F1={best_f1:.4f}, ROC-AUC={roc:.4f}")
            self.logger.info("\n" + classification_report(y, y_pred_full))

            #  MLflow Logging
            input_example = X.head(5)
            mlflow.xgboost.log_model(
                final_model, name="xgboost_final_model", input_example=input_example)
            mlflow.log_metrics(
                {"final_accuracy": acc, "final_f1": best_f1, "final_roc_auc": roc})
            mlflow.log_metric("final_threshold", best_threshold)

            #  Save Preprocessing Artifact
            os.makedirs("artifacts", exist_ok=True)
            preproc_path = f"artifacts/preprocessor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(X, preproc_path)
            mlflow.log_artifact(preproc_path)

            from mlflow.models import infer_signature
            signature = infer_signature(X, final_model.predict(X))
            input_example = X.head(5)

            mlflow.xgboost.log_model(
                final_model,
                name="xgboost_final_model",
                signature=signature,
                input_example=input_example)

            # Log feature names and other relevant metadata
            feature_metadata = {
                "feature_names": X.columns.tolist(),
                "n_features": len(X.columns),
                "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()}
            }

            # Save feature metadata as JSON
            os.makedirs("artifacts", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metadata_path = f"artifacts/feature_metadata_{timestamp}.json"

            try:
                with open(metadata_path, 'w') as f:
                    json.dump(feature_metadata, f, indent=2)
                mlflow.log_artifact(metadata_path, name="feature_metadata")
                self.logger.info(
                    f"Feature metadata saved successfully at {metadata_path}")
            except Exception as e:
                self.logger.error(f"Failed to save feature metadata: {str(e)}")

            return final_model, fold_metrics

    def save_model(self, model):
        MODEL_DIR = os.getenv("MODEL_DIR", "models")
        os.makedirs(MODEL_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            MODEL_DIR, f"xgboost_model_{timestamp}.joblib")
        joblib.dump(model, model_path)

        print(model_path)
        self.logger.info(f"Final model saved locally at {model_path}")


# Main Execution
if __name__ == "__main__":
    config_path = "config/config_train_xgb.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(
        config["logging"]["log_path"], config["logging"]["log_level"])
    logger.info("Loading preprocessed data...")

    if os.path.exists("data/processed/processed_data.csv"):
        df_processed = pd.read_csv("data/processed/processed_data.csv")
    else:
        df_processed = fetch_preprocessed()
    target_col = config["target_column"]
    logger.info(f"Target column: {target_col}")

    # Check if target column exists
    if target_col not in df_processed.columns:
        available_cols = df_processed.columns.tolist()
        logger.error(
            f"Target column '{target_col}' not found. Available columns: {available_cols}")
        raise ValueError(f"Target column '{target_col}' not found in data")

    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    trainer = XGBoostTrainer(config=config, logger=logger)
    best_model, fold_metrics = trainer.train_and_tune_model(X, y)
    trainer.save_model(best_model)
