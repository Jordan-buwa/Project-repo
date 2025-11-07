FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install Python dependencies
COPY requirements.txt .
# Install dependencies with increased timeout and PyPI mirror
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt -i https://pypi.org/simple

# Copy source code and configurations
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories for training artifacts
RUN mkdir -p \
    models \
    artifacts \
    logs \
    data/raw \
    data/processed

# Set environment variables for MLflow and training paths
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data
ENV LOG_LEVEL=INFO
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_EXPERIMENT_NAME=Churn_Training

# Expose MLflow UI port
EXPOSE 5000

# Run training script
CMD ["python3", "src/models/train_xgb.py"]\
