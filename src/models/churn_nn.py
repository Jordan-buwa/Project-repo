import torch
from src.models.tuning.optuna_nn import run_optuna_optimization
from src.models.utils.eval_nn import evaluate_model
from src.models.utils.train_util import train_model
from src.models.train_NN.neural_net import ChurnNN
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import numpy as np
from src.data_pipeline.pipeline_data import fetch_preprocessed
import yaml
import os
from src.data_pipeline.ingest import setup_logger
from dotenv import load_dotenv
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "models/")
df_processed = fetch_preprocessed()


config_path = "config/config_train.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

logger = setup_logger(
    config["logging"]["log_path"], config["logging"]["log_level"])
logger.info("Loading preprocessed data...")


# Features & target
target_col = config["target_column"]
X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run Optuna optimization
logger.info("Starting hyperparameter optimization with Optuna...")
study = run_optuna_optimization(X, y, n_trials=20, device=device)
logger.info(f"Hyperparameter optimization completed!\nBest Hyperparameters: {study.best_params}\nBest AUC-ROC: {study.best_value:.4f}")

print("\nBest Hyperparameters:", study.best_params)
print(f"Best AUC-ROC: {study.best_value:.4f}")

# Train final model with best params
best_params = study.best_params
n_layers = best_params["n_layers"]
n_units = [best_params[f"n_units_{i}"] for i in range(n_layers)]
dropout_rate = best_params["dropout_rate"]
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)

metrics_all = {"AUC": [], "F1": [], "Recall": [], "Precision": [], "Accuracy": []}

logger.info("Training final model with cross-validation...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    X_train_tensor = torch.tensor(X_train_res.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_res.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ChurnNN(input_size=X.shape[1], n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, num_epochs=20, device=device)
    metrics = evaluate_model(model, X_test_tensor, y_test_tensor, device=device)

    for k, v in metrics.items():
        metrics_all[k].append(v)
    print(f"Fold {fold + 1} metrics: {metrics}")

# Print final average results
print("\nFinal Average Metrics:")
logger.info("Final Average Metrics:")
for k, v in metrics_all.items():
    print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    logger.info(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

path = MODEL_DIR + "churn_model_with_optuna.pth"
torch.save(model.state_dict(), path)
print("Model saved.")
