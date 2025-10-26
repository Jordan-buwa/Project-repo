import torch
import torch.nn as nn

class ChurnNN(nn.Module):
    def __init__(self, input_size, n_layers, n_units, dropout_rate):
        super().__init__()
        layers = []
        in_features = input_size

        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_units[i]

        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)








import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import optuna

class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def create_fold_dataloaders(X, y, n_splits=5, batch_size=64, random_state=42):
    """
    Create Stratified K-Fold DataLoaders with SMOTE applied to training folds.

    """

    # Convert to numpy arrays if DataFrames/Series
    X = np.array(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_loaders = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n Fold {fold + 1}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply SMOTE only on the training data
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Create PyTorch datasets
        train_dataset = ChurnDataset(X_train_res, y_train_res)
        val_dataset = ChurnDataset(X_val, y_val)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        fold_loaders.append((train_loader, val_loader))

        print(f"Fold {fold + 1}: Train size={len(train_dataset)}, Val size={len(val_dataset)}")

    return fold_loaders

class ChurnNN(nn.Module):
    def __init__(self, input_size, n_layers, n_units, dropout_rate):
        super(ChurnNN, self).__init__()
        layers = []
        in_features = input_size
        
        # Dynamically create hidden layers
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_units[i]
        
        # Output layer
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
    return avg_loss

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_probs = test_outputs.numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)
        y_test_np = y_test_tensor.numpy().flatten()

    auc = roc_auc_score(y_test_np, test_probs)
    f1 = f1_score(y_test_np, test_preds)
    recall = recall_score(y_test_np, test_preds)
    precision = precision_score(y_test_np, test_preds)
    accuracy = accuracy_score(y_test_np, test_preds)

    return auc, f1, recall, precision, accuracy

def objective(trial):
    # Hyperparameter search space
    n_layers = trial.suggest_int('n_layers', 1, 4)
    n_units = [trial.suggest_int(f'n_units_{i}', 32, 256, step=32) for i in range(n_layers)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42)
    auc_scores = []

    for train_idx, test_idx in skf.split(X, y):
        # Split data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Apply SMOTE
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_resampled.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = ChurnNN(input_size=X_train.shape[1], n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train & Evaluate
        train_model(model, train_loader, criterion, optimizer, num_epochs=20)
        auc, _, _, _, _ = evaluate_model(model, X_test_tensor, y_test_tensor)
        auc_scores.append(auc)

    return np.mean(auc_scores)


# =====================================================
# Step 5: Run Optuna Study
# =====================================================
def run_optuna_optimization(n_trials=20):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study


# =====================================================
# Step 6: Final Cross-Validation with Best Hyperparameters
# =====================================================
def final_evaluation(best_params):
    n_layers = best_params['n_layers']
    n_units = [best_params[f'n_units_{i}'] for i in range(n_layers)]
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']

    metrics = {'AUC': [], 'F1': [], 'Recall': [], 'Precision': [], 'Accuracy': []}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining Fold {fold + 1}/5 with Best Hyperparameters")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        X_train_tensor = torch.tensor(X_train_resampled.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = ChurnNN(input_size=X_train.shape[1], n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train model
        train_model(model, train_loader, criterion, optimizer, num_epochs=20)

        # Evaluate model
        auc, f1, recall, precision, accuracy = evaluate_model(model, X_test_tensor, y_test_tensor)
        print(f"Fold {fold + 1}: AUC={auc:.4f}, F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, Accuracy={accuracy:.4f}")

        # Save metrics
        metrics['AUC'].append(auc)
        metrics['F1'].append(f1)
        metrics['Recall'].append(recall)
        metrics['Precision'].append(precision)
        metrics['Accuracy'].append(accuracy)

    # Summary
    print("\nFinal Average Metrics (Best Hyperparameters):")
    for k, v in metrics.items():
        print(f"  {k}: {np.mean(v):.4f} (±{np.std(v):.4f})")

    torch.save(model.state_dict(), 'churn_model_with_optuna.pth')
    print("\nModel saved as 'churn_model_with_optuna.pth'")