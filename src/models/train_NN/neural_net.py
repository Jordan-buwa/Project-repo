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
    def predict_proba(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(X_test)
        return probabilities.numpy()