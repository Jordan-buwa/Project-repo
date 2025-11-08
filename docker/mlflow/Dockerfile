# docker/mlflow/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install only essential packages for MLflow
RUN pip install --no-cache-dir --timeout=300 \
    mlflow==2.11.3 \
    psycopg2-binary==2.9.7 \
    gunicorn==21.2.0

# Copy application code
COPY . .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1