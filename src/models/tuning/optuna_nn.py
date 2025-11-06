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
import logging
from sklearn.impute import SimpleImputer
import os
from src.models.network.neural_net import ChurnNN
from src.models.utils.train_util import train_model
from src.models.utils.eval_nn import evaluate_model

# --- Load config ---
with open("config/config_train_nn.yaml", "r") as f:
    config = yaml.safe_load(f)

device = config["device"]
num_epochs = config["training"]["num_epochs"]
n_trials = config["training"]["n_trials"]
log_path = config["paths"]["logs"]
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def setup_logger(log_path):
    logger = logging.getLogger("OptunaStudy")
    if logger.handlers:
        return logger
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    
    # Console handler with simpler format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    
    logger.setLevel(logging.INFO)
    
    # Set other loggers to WARNING
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger

# === Optuna Logger ===
class OptunaLogger:
    def __init__(self, logger):
        self.logger = logger
    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.logger.info(f"[Trial {trial.number}] F1: {trial.value:.4f} | {trial.params}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            self.logger.warning(f"[Trial {trial.number}] Pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            self.logger.error(f"[Trial {trial.number}] Failed")
        

optuna_logger = OptunaLogger(setup_logger(log_path))

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
    f1_scores = []

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
        f1_scores.append(metrics["F1"])

    return np.mean(f1_scores)

# --- Run Optuna study ---
def run_optuna_optimization(X, y, n_trials=n_trials, device=device):
    mlflow_callback = MLflowCallback(metric_name="f1_score")
    study = optuna.create_study(direction='maximize', study_name="Churn NN Optimization")
    study.optimize(lambda trial: objective(trial, X, y, device), n_trials=n_trials, callbacks=[optuna_logger, mlflow_callback])
    return study
