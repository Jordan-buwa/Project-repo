import torch.nn as nn
import pandas as pd
import numpy as np
import torch

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
    def predict_proba(self, X_test):
        try:    
            if not isinstance(X_test, torch.Tensor):
                if isinstance(X_test, pd.DataFrame):
                    X_test = X_test.values
                X_test = torch.tensor(X_test, dtype=torch.float32)
                if isinstance(X_test, np.ndarray):
                    X_test = torch.from_numpy(X_test)
            self.eval()
            with torch.no_grad():
                probabilities = self.forward(X_test)
            return probabilities.numpy()
        except Exception as e:
            raise ValueError(f"Error during prediction with neural network: {str(e)}")