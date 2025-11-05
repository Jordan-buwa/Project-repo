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

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY retrain.py .

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
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)" || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data
ENV LOG_LEVEL=INFO

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]