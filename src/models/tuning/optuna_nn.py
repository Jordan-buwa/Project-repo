import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import optuna
import logging
from sklearn.impute import SimpleImputer

from src.models.train_NN.neural_net import ChurnNN
from src.models.utils.train_util import train_model
from src.models.utils.eval_nn import evaluate_model

def objective(trial, X, y, device):
    logger = logging.getLogger(__name__)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    n_units = [trial.suggest_int(f'n_units_{i}', 32, 256, step=32) for i in range(n_layers)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42)
    auc_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        try:
            # ensure imputation before SMOTE to handle NaNs
            impute_strategy = trial.suggest_categorical("impute_strategy", ["median", "mean", "most_frequent"])
            logger.debug(f"Using impute_strategy={impute_strategy} before SMOTE")
            imputer = SimpleImputer(strategy=impute_strategy)

            pipeline = ImbPipeline([("imputer", imputer), ("smote", smote)])

            logger.info("Running imputer + SMOTE pipeline on training data")
            X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
            logger.info(f"Resampled training set: {X_train_res.shape[0]} rows")

        except Exception as e:
            logger.exception("Resampling (imputer+SMOTE) failed")
            raise

        X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_res, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = ChurnNN(input_size=X.shape[1], n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_model(model, train_loader, criterion, optimizer, num_epochs=20, device=device)
        metrics = evaluate_model(model, X_test_tensor, y_test_tensor, device=device)
        auc_scores.append(metrics["AUC"])

    return np.mean(auc_scores)


def run_optuna_optimization(X, y, n_trials=20, device='cpu'):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, device), n_trials=n_trials)
    return study
