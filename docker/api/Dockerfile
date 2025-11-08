FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including PostgreSQL client and curl for health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for better caching)
COPY requirements.txt .
#COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -e .

# Install additional Azure packages
RUN pip install --no-cache-dir \
    azure-storage-blob \
    azure-identity \
    opencensus-ext-azure \
    opencensus-ext-logging

# Copy project files
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY data/ ./data/
COPY artifacts/ ./artifacts/

# Create necessary directories
RUN mkdir -p \
    logs \
    models \
    artifacts \
    data/processed \
    data/raw \
    data/ingestion \
    src/api/cache \
    src/api/utils/logs

# Expose API port
EXPOSE 8000

# Health check (fixed - uses wget instead of curl)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app/src:/app
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data
ENV LOG_LEVEL=INFO
ENV POSTGRES_HOST=localhost
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB_NAME=churn_db
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]