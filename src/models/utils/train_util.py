import torch
from src.data_pipeline.ingest import setup_logger
import os
import mlflow
import logging
from datetime import datetime

def train_model(model, train_loader, criterion, optimizer, num_epochs=20, device='cpu', model_name = "Neural_Network"):
    model.to(device)
    model.train()
    os.makedirs("src/models/utils/logs", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"src/models/utils/logs/train_{timestamp}.log"
    logging = setup_logger(path, "INFO")
    logging.info(f"Starting training for {model_name} for {num_epochs} epochs on {device}")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return model, avg_loss
