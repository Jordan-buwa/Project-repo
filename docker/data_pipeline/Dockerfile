FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and set timeout
RUN pip install --upgrade pip
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

# Copy requirements
COPY requirements.txt .

# Install in layers - most stable packages first
# Layer 1: Core data science (rarely change)
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    pandas \
    numpy \
    scikit-learn \
    scipy

# Layer 2: Configuration and database  
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    pyyaml \
    python-dotenv \
    sqlalchemy \
    psycopg2-binary

# Layer 3: ML tools (optional)
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    mlflow

# Layer 4: Everything else (changes most frequently)
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

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

# Use main.py directly
CMD ["python", "src/data_pipeline/main.py", "--help"]