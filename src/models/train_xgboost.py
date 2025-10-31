from collections import defaultdict
from src.data_pipeline.preprocess import DataPreprocessor
import os
import yaml
import joblib
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.combine import SMOTETomek
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.xgboost
from src.data_pipeline.pipeline_data import fetch_preprocessed
import warnings
import subprocess

[warnings.filterwarnings("ignore", category=c)
 for c in (UserWarning, FutureWarning)]


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
    return logging.getLogger(__name__)


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

        # MLflow setup
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("XGBoost_Churn_Experiment")
        self.logger.info(f"MLflow tracking URI: {mlflow_uri}")

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

                xgb = XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    random_state=self.config["random_state"]
                )

                tuner = RandomizedSearchCV(
                    estimator=xgb,
                    param_distributions=param_grid,
                    scoring='f1',
                    n_iter=10,
                    cv=3,
                    n_jobs=-1,
                    random_state=self.config["random_state"]
                )
                tuner.fit(X_train, y_train)
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

                fold_signature = infer_signature(
                    X_val, best_model.predict(X_val))
                fold_input_example = X_val.head(5)
                mlflow.xgboost.log_model(best_model,
                                         name=f"xgboost_model_fold_{fold}",
                                         signature=fold_signature,
                                         input_example=fold_input_example)

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
                smt = SMOTETomek(random_state=self.config["random_state"])
                X, y = smt.fit_resample(X, y)
                self.logger.info(
                    f"Full dataset size after SMOTETomek: {X.shape[0]}")

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
            signature = infer_signature(X, final_model.predict(X))
            input_example = X.head(5)
            mlflow.xgboost.log_model(final_model,
                                     name="xgboost_final_model",
                                     signature=signature,
                                     input_example=input_example)
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

            # Remove the preprocessing artifacts saving block since we're not using self.dp
            # Instead, log feature names and other relevant metadata
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
                self.logger.info(f"Feature metadata saved successfully at {metadata_path}")
            except Exception as e:
                self.logger.error(f"Failed to save feature metadata: {str(e)}")
                
            return final_model, fold_metrics

    def save_model(self, model):
        MODEL_DIR = os.getenv("MODEL_DIR", "models")
        os.makedirs(MODEL_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"xgboost_model_{timestamp}.joblib")
        joblib.dump(model, model_path)

        print(model_path) 
        self.logger.info(f"Final model saved locally at {model_path}")



# Main Execution
if __name__ == "__main__":
    config_path = "config/config_train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(
        config["logging"]["log_path"], config["logging"]["log_level"])
    logger.info("Loading preprocessed data...")

    df_processed = fetch_preprocessed()
    target_col = config["target_column"]
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    trainer = XGBoostTrainer(config=config, logger=logger)
    best_model, fold_metrics = trainer.train_and_tune_model(X, y)
    trainer.save_model(best_model)
