import torch
from src.models.tuning.optuna_nn import run_optuna_optimization, optuna_logger
from src.models.utils.eval_nn import evaluate_model
from src.models.utils.train_util import train_model
from src.models.train_NN.neural_net import ChurnNN
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.data_pipeline.pipeline_data import fetch_preprocessed
import yaml
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import os
from datetime import datetime
from src.data_pipeline.ingest import setup_logger
from dotenv import load_dotenv
import warnings
# Suppressing unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Load environment variables
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
os.makedirs(MODEL_DIR, exist_ok=True)

config_path = "config/config_train_nn.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
df_processed = fetch_preprocessed()

logger = optuna_logger.logger
logger.info("Loading preprocessed data...")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Neural network for Churn model")
logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
# Features & target
target_col = config["target_column"]
X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
# Run Optuna optimization
logger.info("Starting hyperparameter optimization with Optuna...")
study = run_optuna_optimization(X, y, n_trials=20, device=device)
logger.info(f"Hyperparameter optimization completed!\nBest Hyperparameters: {study.best_params}\nBest AUC-ROC: {study.best_value:.4f}")

print("\nBest Hyperparameters:", study.best_params)
print(f"Best F1-score: {study.best_value:.4f}")

# Train final model with best params
best_params = study.best_params
n_layers = best_params["n_layers"]
n_units = [best_params[f"n_units_{i}"] for i in range(n_layers)]
dropout_rate = best_params["dropout_rate"]
learning_rate = best_params["learning_rate"]
batch_size = best_params["batch_size"]
mlflow.log_params(study.best_params)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with mlflow.start_run(run_name=f"NN_training_{timestamp}", nested=True):
    script_name = os.path.basename(__file__) if "__file__" in globals() else "notebook"
    mlflow.set_tag("script_version", script_name)
    mlflow.log_param("num_samples", X.shape[0])
    mlflow.log_param("num_features", X.shape[1])

    # Log hyperparameters
    mlflow.log_params(best_params)

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
        disp, metrics = evaluate_model(model, X_test_tensor, y_test_tensor, device=device)
        plt.savefig("confusion_matrix.png")

        mlflow.log_artifact("confusion_matrix.png")
        for k, v in metrics.items():
            metrics_all[k].append(v)
            mlflow.log_metric(f"{k}_fold_{fold+1}", v)
        print(f"Fold {fold + 1} metrics: {metrics}")
        logger.info(f"Fold {fold + 1} metrics: {metrics}")
    # Log average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics_all.items()}
    mlflow.log_metrics(avg_metrics)
    # Print final average results
    print("\nFinal Average Metrics:")
    logger.info("Final Average Metrics:")
    for k, v in metrics_all.items():
        print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
        logger.info(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"nn_model_{timestamp}.pth")

    model.eval()
    # Prepare input
    if isinstance(X, pd.DataFrame):
        input_tensor = torch.tensor(X.values, dtype=torch.float32)
        input_example = X.head(3).to_dict(orient="records")
    else:
        input_tensor = torch.tensor(X, dtype=torch.float32)
        input_example = X[:3].tolist()

    # Predict on sample
    with torch.no_grad():
        sample_pred = model(input_tensor[:3]).cpu().numpy()

    # Infer signature
    signature = infer_signature(X.head(3) if isinstance(X, pd.DataFrame) else X[:3], sample_pred)

    # Log with MLflow
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name=f"churn_model_{timestamp}",
        signature=signature,
        input_example=input_example
    )

    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(f"nn_model_{timestamp}.pth", "nn_churn_model")
    print(f"Model saved at {model_path}")
    logger.info(f"Model saved at {model_path}")
    # Register model
    try:
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", "nn_churn_model")
        logger.info(f"Model registered in MLflow Registry as 'nn_churn_model'")
    except Exception as e:
        logger.warning(f"Failed to register model: {e}")
    

