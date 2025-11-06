import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import optuna
from optuna.integration import MLflowCallback
import mlflow
import logging
from sklearn.impute import SimpleImputer
import os
from src.models.network.neural_net import ChurnNN
from src.models.utils.train_util import train_model
from src.models.utils.eval_nn import evaluate_model
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)
# --- Load config ---
with open("config/config_train_nn.yaml", "r") as f:
    config = yaml.safe_load(f)

device = config["device"]
num_epochs = config["training"]["num_epochs"]
n_trials = config["training"]["n_trials"]

class DerivedMLflowCallback(MLflowCallback):
    """Custom MLflow callback that logs all metrics quietly"""
    
    def __init__(self, metric_names=None, metric_name="f1_score"):
        super().__init__(metric_name=metric_name)
        self.metric_names = metric_names or ["f1_score", "accuracy", "roc_auc", "precision", "recall"]
    
        import mlflow
        mlflow.set_experiment("Neuralnet_Churn_Experiment")
    
    def __call__(self, study, trial):
        # Completely suppress all output during MLflow operations
        import logging
        mlflow_logger = logging.getLogger("mlflow")
        original_level = mlflow_logger.getEffectiveLevel()
        mlflow_logger.setLevel(logging.WARNING)
        
        try:
            # Call parent functionality
            super().__call__(study, trial)
            
            # Add our custom metrics for completed trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs:
                for key, value in trial.user_attrs.items():
                    if any(metric in key for metric in self.metric_names):
                        mlflow.log_metric(key, value)
        finally:
            # Restore original logging level
            mlflow_logger.setLevel(original_level)


# Optuna Logger
class OptunaLogger():
    def __init__(self, logger):
        self.logger = logger
        
    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.logger.info(f"[Trial {trial.number}] F1: {trial.value:.4f} | Params: {list(trial.params.keys())}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            self.logger.warning(f"[Trial {trial.number}] Pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            self.logger.error(f"[Trial {trial.number}] Failed: {trial.user_attrs.get('error', 'Unknown error')}")

optuna_logger = OptunaLogger(logging.getLogger("optuna"))


# --- Objective function ---
def objective(trial, X, y, device=device):
    n_layers = trial.suggest_int('n_layers', config["optuna"]["n_layers"]["min"], config["optuna"]["n_layers"]["max"])
    n_units = [trial.suggest_int(f'n_units_{i}',
                                 config["optuna"]["n_units"]["min"],
                                 config["optuna"]["n_units"]["max"],
                                 step=config["optuna"]["n_units"]["step"])
               for i in range(n_layers)]
    drop_min = config["optuna"]["dropout_rate"]["min"]
    drop_max = config["optuna"]["dropout_rate"]["max"]
    dropout_rate = trial.suggest_float('dropout_rate',
                                       drop_min,
                                       drop_max)
    lr_min = config["optuna"]["learning_rate"]["min"]
    lr_max = config["optuna"]["learning_rate"]["max"]
    learning_rate = trial.suggest_float('learning_rate',
                                        lr_min,
                                        lr_max,
                                        log=True)
    batch_size = trial.suggest_categorical('batch_size', config["optuna"]["batch_size"])
    impute_strategy = trial.suggest_categorical("impute_strategy", config["optuna"]["impute_strategy"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42)
    # Store all metrics
    all_metrics = {
        "f1_scores": [], "auc_scores": [], "accuracy_scores": [],
        "precision_scores": [], "recall_scores": []
    }

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Impute + SMOTE
        imputer = SimpleImputer(strategy=impute_strategy)
        pipeline = ImbPipeline([("imputer", imputer), ("smote", smote)])
        X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_res, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Model
        model = ChurnNN(input_size=X.shape[1], n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
        _, metrics = evaluate_model(model, X_test_tensor, y_test_tensor, device=device)
        # Collect all metrics
        all_metrics["f1_scores"].append(metrics["F1"])
        all_metrics["auc_scores"].append(metrics.get("AUC", 0))
        all_metrics["accuracy_scores"].append(metrics.get("Accuracy", 0))
        all_metrics["precision_scores"].append(metrics.get("Precision", 0))
        all_metrics["recall_scores"].append(metrics.get("Recall", 0))
    
    # Calculate mean metrics
    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    # Store additional metrics as user attributes
    trial.set_user_attr("mean_auc", mean_metrics["auc_scores"])
    trial.set_user_attr("mean_accuracy", mean_metrics["accuracy_scores"])
    trial.set_user_attr("mean_precision", mean_metrics["precision_scores"])
    trial.set_user_attr("mean_recall", mean_metrics["recall_scores"])
    
    # Return F1 as primary objective
    return mean_metrics["f1_scores"]

# --- Run Optuna study ---
def run_optuna_optimization(X, y, n_trials=n_trials, device=device):
    mlflow_callback = DerivedMLflowCallback(metric_name="f1_score")
    study = optuna.create_study(direction='maximize', study_name="Churn NN Optimization")
    study.optimize(lambda trial: objective(trial, X, y, device), n_trials=n_trials, callbacks=[optuna_logger, mlflow_callback])
    return study
