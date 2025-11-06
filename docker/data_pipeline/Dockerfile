# Dockerfile for Data Pipeline
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p src/data_pipeline/logs/ingested \
    src/data_pipeline/logs/preprocessed \
    data/backup \
    data/processed \
    data/ingestion/failed_batches

# Set environment variables
ENV PYTHONPATH=/app

# Run data pipeline
CMD ["python", "src/data_pipeline/data_pipeline.py"]