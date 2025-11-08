from src.models.tuning.optuna_nn import run_optuna_optimization
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from src.data_pipeline.pipeline_data import fetch_preprocessed
from src.models.utils.util_nn import create_fold_dataloaders
from src.models.utils.eval_nn import evaluate_model
from src.models.utils.train_util import train_model
from src.models.network.neural_net import ChurnNN

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from mlflow.models import infer_signature
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import mlflow.pytorch
import numpy as np
import pandas as pd
import subprocess
import warnings
import os, sys
import logging
import mlflow
import torch
import yaml

# Suppressing unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(str(Path(__file__).parent.parent))
# Load environment variables
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "models/")
os.makedirs(MODEL_DIR, exist_ok=True)
#os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

config_path = "config/config_train_nn.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
logger = logging.getLogger(__name__)

class NeuralNetworkTrainer():
    def __init__(self, X, y, config, device, MODEL_DIR=MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.X = X
        self.y = y
        self.config = config
        self.device = device
        self.best_params = None
        self.logger = logger
        self.random_state = self.config["training"]["random_state"]
        self.model = None
        self.num_splits_cv = self.config["training"]["num_splits_cv"]
        self.n_trials = self.config["training"]["n_trials"]
        self.num_epochs = self.config["training"]["num_epochs"]
        self.batch_size = self.config["optuna"]["batch_size"][0]
        self.learning_rate = None
        self.n_layers = None
        self.n_units = None
        self.dropout_rate = None
        self.best_params = None
        self.dvc_hash = None
        self.smote = SMOTE(random_state=self.random_state)
        self.skf = StratifiedKFold(n_splits=self.num_splits_cv,
                                    shuffle=True, 
                                    random_state=self.random_state)
        self.criterion = torch.nn.BCELoss()

    def create_fold_dataloaders(self):
        return create_fold_dataloaders(self.X, self.y, self.num_splits_cv, self.batch_size, self.random_state)
    
    def tuner(self):
        return run_optuna_optimization(self.X, self.y, self.n_trials, self.device)
    def train_model(self, train_loader):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return train_model(self.model, train_loader, self.criterion, self.optimizer, self.num_epochs, self.device)
    def evaluate_model(self, X_test_tensor, y_test_tensor):
        return evaluate_model(self.model, X_test_tensor, y_test_tensor, self.device)

    
    def get_prediction_threshold(self, y_true, y_probs):
        best_thresh, best_f1 = 0.5, 0
        for t in [i * 0.01 for i in range(1, 100)]:
            preds = (y_probs >= t).astype(int)
            score = f1_score(y_true, preds)
            if score > best_f1:
                best_f1 = score
                best_thresh = t
        return best_thresh, best_f1
    def train_and_tune(self):
        # Run Optuna optimization
        self.logger.info("Starting hyperparameter optimization with Optuna...")
        study = self.tuner()
        self.logger.info(f"Hyperparameter optimization completed!\nBest Hyperparameters: {study.best_params}\nBest AUC-ROC: {study.best_value:.4f}")
        print("\nBest Hyperparameters:", study.best_params)
        print(f"Best F1-score: {study.best_value:.4f}")
        self.best_params = study.best_params
        self.batch_size = self.best_params["batch_size"]

        # Train final model with best params
        self.n_layers = self.best_params["n_layers"]
        self.n_units = [self.best_params[f"n_units_{i}"] for i in range(self.n_layers)]
        self.dropout_rate = self.best_params["dropout_rate"]
        self.learning_rate = self.best_params["learning_rate"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_metrics = []
        # Fetch DVC hash for tracking
        try:
            self.dvc_hash = subprocess.getoutput(
                "dvc hash data/processed/preprocessed.csv")
        except Exception:
            self.dvc_hash = "N/A"

        # MLflow setup
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("Neuralnet_Churn_Experiment")
        self.logger.info(f"MLflow tracking URI: {mlflow_uri}")
        with mlflow.start_run():
            script_name = os.path.basename(__file__) if "__file__" in globals() else "notebook"
            mlflow.set_tag("script_version", script_name)
            mlflow.log_param("num_samples", X.shape[0])
            mlflow.log_param("num_features", X.shape[1])

            # Log hyperparameters
            mlflow.log_params(self.best_params)

            metrics_all = {"AUC": [], "F1": [], "Recall": [], "Precision": [], "Accuracy": []}
            y_true_global = []
            y_pred_global = []
            self.logger.info("Training final model with cross-validation...")
            for fold, (train_idx, test_idx) in enumerate(self.skf.split(X, y)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
                X_train_res, y_train_res = self.smote.fit_resample(X_train, y_train)

                X_train_tensor = torch.tensor(X_train_res.values, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train_res.values, dtype=torch.float32)
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

                self.model = ChurnNN(input_size=X.shape[1], n_layers=self.n_layers, n_units=self.n_units, dropout_rate=self.dropout_rate)

                self.model, _ = self.train_model(train_loader)
                disp, metrics = self.evaluate_model(X_test_tensor, y_test_tensor)
                # Plot confusion matrix
                _, ax = plt.subplots(figsize=(6, 6))
                disp.plot(ax=ax)
                plt.savefig("images/confusion_matrix.png")
                mlflow.log_artifact("images/confusion_matrix.png")
                # Predict probabilities and determine best threshold
                self.logger.info(f"Shape of X_test: {X_test.shape}\nShape of model output: {self.model.predict_proba(X_test).shape}")
                y_probs = self.model.predict_proba(X_test)[:, 1]
                best_threshold, best_f1 = self.get_prediction_threshold(y_test, y_probs)
                y_pred = (y_probs >= best_threshold).astype(int)

                acc = accuracy_score(y_test, y_pred)
                roc = roc_auc_score(y_test, y_probs)
                y_true_global.extend(y_test.tolist())
                y_pred_global.extend(y_pred.tolist())
                self.logger.info(
                    f"Fold {fold}: Accuracy={acc:.4f}, F1={best_f1:.4f}, ROC-AUC={roc:.4f}")

                mlflow.log_metrics({
                    f"fold_{fold}_accuracy": acc,
                    f"fold_{fold}_f1": best_f1,
                    f"fold_{fold}_roc_auc": roc,
                })
                #  MLflow Model Logging
                fold_input_example = X_test.head(5)
                mlflow.pytorch.log_model(
                    self.model, name=f"nn_model_fold_{fold}", input_example=fold_input_example)

                fold_metrics.append({
                    "fold": fold,
                    "best_params": self.best_params,
                    "accuracy": acc,
                    "f1_score": best_f1,
                    "roc_auc": roc,
                    "threshold": best_threshold,
                    "y_val_list": y_test.tolist(),
                    "y_pred_list": y_pred.tolist()
                })

                for k, v in metrics.items():
                    metrics_all[k].append(v)
                    mlflow.log_metric(f"{k}_fold_{fold+1}", v)
                print(f"Fold {fold + 1} metrics: {metrics}")
                self.logger.info(f"Fold {fold + 1} metrics: {metrics}")

            # Log average metrics
            avg_metrics = {k: np.mean(v) for k, v in metrics_all.items()}
            mlflow.log_metrics(avg_metrics)
            # Print final average results
            print("\nFinal Average Metrics:")
            self.logger.info("Final Average Metrics:")
            for k, v in metrics_all.items():
                print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
                self.logger.info(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
            # Compute Global F1
            global_f1 = f1_score(
                y_true_global, y_pred_global, average="binary")
            self.logger.info(f"Global F1 across all folds: {global_f1:.4f}")
            mlflow.log_metric("global_f1", global_f1)
            y_probs_full = self.model.predict_proba(X)[:, 1]
            best_threshold, best_f1 = self.get_prediction_threshold(y, y_probs_full)
            y_pred_full = (y_probs_full >= best_threshold).astype(int)

            acc = accuracy_score(y, y_pred_full)
            roc = roc_auc_score(y, y_probs_full)
            self.logger.info(
                f"Final model: Accuracy={acc:.4f}, F1={best_f1:.4f}, ROC-AUC={roc:.4f}")
            self.logger.info("\n" + classification_report(y, y_pred_full))
        return self
        
    # Save model with timestamp
    def save_model(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mod_name = f"neural_net_model_{timestamp}.pth"
        model_path = os.path.join(MODEL_DIR, mod_name)
        try:
            torch.save(self.model.state_dict(), model_path)
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            print(f"Error saving model: {e}")
            raise
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        mlflow.log_artifact(mod_name, artifact_path="nn_churn_model", registered_model_name="nn_churn_model", run_id=mlflow.active_run().info.run_id)
        print(f"Model saved at {model_path}")
        self.logger.info(f"Model saved at {model_path}")
        return model_path
    def register_model(self):
        # Register model
        try:
            run_id = mlflow.active_run().info.run_id
            mlflow.register_model(f"runs:/{run_id}/model", "nn_churn_model")
            self.logger.info(f"Model registered in MLflow Registry as 'nn_churn_model'")
        except Exception as e:
            self.logger.warning(f"Failed to register model: {e}")
        return self
    def save_logs_on_local(self, path="src/data_pipeline/training/logs/"):
        # Save training logs
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(path, f"nn_training_log_{timestamp}.log")
        try:
            with open(log_file, "w") as f:
                for handler in self.logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.stream.seek(0)
                        f.write(handler.stream.read())
            self.logger.info(f"Training logs saved at {log_file}")
            mlflow.log_artifact(log_file, artifact_path="training_logs", run_id=mlflow.active_run().info.run_id)
        except Exception as e:
            self.logger.error(f"Error saving training logs: {e}")
        return self
        
    
if __name__ == "__main__":
    if os.path.exists("data/processed/processed_data.csv"):
        df_processed = pd.read_csv("data/processed/processed_data.csv")
    else: df_processed = fetch_preprocessed()
    # Features & target
    target_col = config["target_column"]
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = NeuralNetworkTrainer(X, y, config, device)
    trainer.train_and_tune().save_model()
    trainer.register_model() 
    trainer.save_logs_on_local()