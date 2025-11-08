FROM python:3.9-slim

WORKDIR /app

# Installing system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copying requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installing production dependencies
RUN pip install --no-cache-dir \
    gunicorn==21.2.0 \
    psycopg2-binary==2.9.7

# Copying application code
COPY . .

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1