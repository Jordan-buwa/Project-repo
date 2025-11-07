FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create directory structure
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/backup \
    /app/data/ingestion/failed_batches \
    /app/logs/ingested \
    /app/logs/preprocessed \
    /app/logs/validation

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV LOG_DIR=/app/logs

# Use main.py directly - NO ENTRYPOINT!
CMD ["python", "src/data_pipeline/main.py", "--help"]