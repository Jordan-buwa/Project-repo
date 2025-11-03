import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import numpy as np

class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_fold_dataloaders(X, y, n_splits=5, batch_size=64, random_state=42):
    """Create Stratified K-Fold DataLoaders with SMOTE for training folds."""
    X, y = np.array(X), np.array(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_loaders = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        train_dataset = ChurnDataset(X_train_res, y_train_res)
        val_dataset = ChurnDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        fold_loaders.append((train_loader, val_loader))
        print(f"Train size={len(train_dataset)}, Val size={len(val_dataset)}")

    return fold_loaders
