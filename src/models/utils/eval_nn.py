import torch
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

def evaluate_model(model, X_test_tensor, y_test_tensor, device='cpu'):
    model.eval()
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor).cpu().numpy().flatten()
    preds = (outputs > 0.5).astype(int)
    y_true = y_test_tensor.cpu().numpy().flatten()

    return {
        "AUC": roc_auc_score(y_true, outputs),
        "F1": f1_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "Precision": precision_score(y_true, preds),
        "Accuracy": accuracy_score(y_true, preds)
    }
