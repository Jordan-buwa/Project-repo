FROM python:3.9-slim

WORKDIR /app

# Installing system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copying requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying application code
COPY src/ ./src/
COPY config/ ./config/

# Replace with:
RUN echo '#!/bin/bash\nset -e\necho "=== Data Pipeline Entrypoint ==="\necho "Creating log directories..."\nmkdir -p /app/logs/ingested /app/logs/preprocessed /app/logs/validation\necho "Running data pipeline with arguments: $@"\nexec python src/data_pipeline/pipeline_data.py "$@"' > /entrypoint.sh
RUN chmod +x /entrypoint.sh
# Create necessary directories for data and logs
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/backup \
    /app/data/ingestion/failed_batches \
    /app/logs/ingested \
    /app/logs/preprocessed \
    /app/logs/validation

# Setting environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV LOG_DIR=/app/logs

# Using entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]