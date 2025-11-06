import yaml
import os
import logging
import pickle
from datetime import datetime
import subprocess
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
import mlflow
import mlflow.sklearn

from src.data_pipeline.pipeline_data import fetch_preprocessed
from src.data_pipeline.preprocess import DataPreprocessor


# LOGGER SETUP
def setup_logger(log_path: str, log_level: str = "INFO"):
    """Create a logger that writes to file with timestamped filename."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_path = log_path.replace(".log", f"_{timestamp}.log")
    logging.basicConfig(
        filename=full_log_path,
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


# RANDOM FOREST TRAINER
class RandomForestTrainer:
    def __init__(self, config: dict, logger: logging.Logger, df_processed=None):
        self.config = config
        self.logger = logger

        print("Initializing RandomForestTrainer...")

        # Load MLflow settings
        self.MLFLOW_URI = (
            config.get("mlflow", {}).get("tracking_uri")
            or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_tracking_uri(self.MLFLOW_URI)

        self.experiment_name = (
            config.get("mlflow", {}).get("experiment_name")
            or os.getenv("MLFLOW_EXPERIMENT_NAME", "Customer Churn Model Training")
        )
        mlflow.set_experiment(self.experiment_name)
        self.logger.info(f"MLflow URI: {self.MLFLOW_URI}, Experiment: {self.experiment_name}")
        print(f"MLflow tracking set: {self.MLFLOW_URI} | Experiment: {self.experiment_name}")

        # fetching preprocessed data if not provided
        if df_processed is None:
            print("📦 Fetching preprocessed data...")
            df_processed = fetch_preprocessed()
            print(f" Data fetched: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")

        # Cleaning string columns: strip + lower
        print("🧹 Cleaning string columns...")
        str_cols = df_processed.select_dtypes(include="object").columns
        for col in str_cols:
            df_processed[col] = df_processed[col].str.strip().str.lower()

        self.dp = DataPreprocessor(data_raw=df_processed)
        print("Data preprocessing initialized.")

    @staticmethod
    def find_best_threshold(y_true, y_probs, threshold_cfg=None):
        """Find threshold that maximizes F1 score."""
        start = threshold_cfg.get("start", 0.01) if threshold_cfg else 0.01
        end = threshold_cfg.get("end", 0.99) if threshold_cfg else 0.99
        step = threshold_cfg.get("step", 0.01) if threshold_cfg else 0.01
        thresholds = [start + i * step for i in range(int((end - start) / step) + 1)]
        best_thresh, best_f1 = 0.5, 0
        for t in thresholds:
            preds = (y_probs >= t).astype(int)
            score = f1_score(y_true, preds)
            if score > best_f1:
                best_f1 = score
                best_thresh = t
        return best_thresh, best_f1

    # TRAINING AND TUNING
    def train_and_tune(self, X, y):
        """Train and tune Random Forest using Stratified K-Fold + RandomizedSearchCV."""
        print(" Starting Random Forest training and tuning...")
        model_cfg = self.config["model_selection"]
        param_distributions = self.config["hyperparameters"].get("random_forest", {})
        thresholds_cfg = model_cfg.get("threshold_search", {})
        random_state = self.config.get("model", {}).get("random_state", 42)

        # Stratified CV split
        skf = StratifiedKFold(
            n_splits=self.config["cv"].get("n_splits", 5),
            shuffle=True,
            random_state=random_state
        )

        dvc_hash = subprocess.getoutput("dvc hash data/processed/preprocessed.csv")
        print(f" DVC data hash: {dvc_hash}")

        best_models = {}
        run_id = None

        for name in model_cfg.get("model_choice", []):
            if name != "random_forest":
                self.logger.warning(f"Model {name} not implemented, skipping.")
                continue

            print(f"\n Training model: {name}")
            model = RandomForestClassifier(random_state=random_state, n_jobs=1)
            tuner = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=self.config["cv"].get("n_iter", 20),
                scoring=self.config["cv"].get("scoring", "roc_auc"),
                cv=3,
                n_jobs=1,
                verbose=self.config["cv"].get("verbose", 1),
                random_state=random_state
            )

            with mlflow.start_run(run_name=f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.set_tag("dvc_data_hash", dvc_hash)
                mlflow.log_param("num_samples", X.shape[0])
                mlflow.log_param("num_features", X.shape[1])
                print("MLflow run started and parameters logged.")

                fold_metrics = []
                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                    print(f"\n Fold {fold}/{skf.get_n_splits()} ...")

                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    if self.config.get("resampling", {}).get("apply_smotetomek", True):
                        print("Applying SMOTETomek resampling...")
                        smt = SMOTETomek(random_state=random_state)
                        X_train, y_train = smt.fit_resample(X_train, y_train)
                        print(f"After resampling: {X_train.shape[0]} samples")

                    print("Running hyperparameter search...")
                    tuner.fit(X_train, y_train)
                    print("Best params found.")

                    best_model = tuner.best_estimator_
                    y_probs = best_model.predict_proba(X_val)[:, 1]
                    best_threshold, best_f1 = self.find_best_threshold(y_val, y_probs, thresholds_cfg)
                    y_pred = (y_probs >= best_threshold).astype(int)

                    metrics = {
                        "accuracy": accuracy_score(y_val, y_pred),
                        "precision": precision_score(y_val, y_pred),
                        "recall": recall_score(y_val, y_pred),
                        "f1": best_f1,
                        "roc_auc": roc_auc_score(y_val, y_pred)
                    }
                    fold_metrics.append(metrics)
                    mlflow.log_metrics({f"fold_{fold}_{k}": v for k, v in metrics.items()})
                    print(f"Fold {fold} metrics: {metrics}")

                mlflow.log_params(tuner.best_params_)
                avg_metrics = {k: np.nanmean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
                mlflow.log_metrics(avg_metrics)

                print(f"\n Average cross-validation metrics: {avg_metrics}")
                best_models[name] = (best_model, avg_metrics)
                run_id = mlflow.active_run().info.run_id

        print("\n Finalizing model training on full dataset...")
        primary_metric = model_cfg.get("primary_metric", "f1")
        best_model_name = max(best_models, key=lambda n: best_models[n][1][primary_metric])
        final_model = best_models[best_model_name][0]
        final_model.fit(X, y)

        os.makedirs(self.config.get("output", {}).get("model_dir", "models"), exist_ok=True)
        model_path = os.path.join(self.config["output"]["model_dir"], f"{best_model_name}_final.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)

        mlflow.sklearn.log_model(final_model, artifact_path="final_model", input_example=X.iloc[:5])
        print(f"Final model saved: {model_path}")
        print("Training completed successfully.")

        return final_model, fold_metrics, run_id


# MAIN EXECUTION
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    def load_config(path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        random_state = cfg.get("model", {}).get("random_state")
        if isinstance(random_state, str):
            if random_state.startswith("${") and random_state.endswith("}"):
                env_var = random_state[2:-1]
                cfg["model"]["random_state"] = int(os.getenv(env_var, 42))
            elif random_state.isdigit():
                cfg["model"]["random_state"] = int(random_state)
        elif random_state is None:
            cfg["model"]["random_state"] = int(os.getenv("RANDOM_STATE", 42))
        return cfg

    print("Loading configuration file...")
    config_path = "config/config_train_rf.yaml"
    config = load_config(config_path)
    print("Configuration loaded successfully.")

    logger = setup_logger(config["logging"]["log_path"], config["logging"]["log_level"])
    print("Logger initialized.")

    print("Fetching preprocessed data...")
    df_processed = fetch_preprocessed()
    print(f" Data loaded: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")

    str_cols = df_processed.select_dtypes(include="object").columns
    for col in str_cols:
        df_processed[col] = df_processed[col].str.strip().str.lower()

    target_col = config["data"]["target_column"]
    if target_col not in df_processed.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    print("Starting training pipeline...")
    trainer = RandomForestTrainer(config=config, logger=logger, df_processed=df_processed)
    best_model, fold_metrics, run_id = trainer.train_and_tune(X, y)
    print(" Random Forest training completed successfully.")
