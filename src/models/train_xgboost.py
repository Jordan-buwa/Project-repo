import os
import yaml
import json
import joblib
import logging
from datetime import datetime, timezone
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.combine import SMOTETomek
import mlflow
import mlflow.xgboost
from src.data_pipeline.pipeline_data import fetch_preprocessed

# Import DataPreprocessor
from src.data_pipeline.preprocess import DataPreprocessor

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


# XGBoost Trainer with MLflow

class XGBoostTrainer:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

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

        # Start MLflow run
        mlflow.set_experiment("XGBoost_Churn_Experiment")
        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:

            # Log config
            mlflow.log_params(self.config)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                self.logger.info(f"Starting fold {fold}...")

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

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

                acc = accuracy_score(y_val, y_pred)
                roc = roc_auc_score(y_val, y_pred)

                self.logger.info(
                    f"Fold {fold}: Accuracy={acc:.4f}, F1={best_f1:.4f}, ROC-AUC={roc:.4f}")

                # Log fold metrics & model
                mlflow.log_metrics({
                    f"fold_{fold}_accuracy": acc,
                    f"fold_{fold}_f1": best_f1,
                    f"fold_{fold}_roc_auc": roc,
                })
                mlflow.xgboost.log_model(
                    best_model, artifact_path=f"xgboost_model_fold_{fold}")

                fold_metrics.append({
                    "fold": fold,
                    "best_params": tuner.best_params_,
                    "accuracy": acc,
                    "f1_score": best_f1,
                    "roc_auc": roc,
                    "threshold": best_threshold
                })

            # Final model training

            best_fold = max(fold_metrics, key=lambda x: x["f1_score"])
            best_params = best_fold["best_params"]
            self.logger.info(
                f"Best fold: {best_fold['fold']} with F1={best_fold['f1_score']:.4f}")

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
            roc = roc_auc_score(y, y_pred_full)
            self.logger.info(
                f"Final model performance: Accuracy={acc:.4f}, F1={best_f1:.4f}, ROC-AUC={roc:.4f}")
            self.logger.info("\n" + classification_report(y, y_pred_full))

            # Log final model & metrics
            mlflow.log_metrics(
                {"final_accuracy": acc, "final_f1": best_f1, "final_roc_auc": roc})
            mlflow.xgboost.log_model(
                final_model, artifact_path="xgboost_final_model")

            # Save preprocessing artifacts
            os.makedirs("artifacts", exist_ok=True)
            preproc_path = f"artifacts/preprocessor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(self.dp, preproc_path)
            mlflow.log_artifact(preproc_path, artifact_path="preprocessing")

            return final_model, fold_metrics


# Main Execution
if __name__ == "__main__":
    config_path = "config/config_train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(
        config["logging"]["log_path"], config["logging"]["log_level"])
    logger.info("Loading preprocessed data...")

    df_processed = fetch_preprocessed()

    # Features & target
    target_col = config["target_column"]
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    # Train and log model
    trainer = XGBoostTrainer(config=config, logger=logger)
    best_model, fold_metrics = trainer.train_and_tune_model(X, y)

    # end
