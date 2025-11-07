#!/bin/bash
set -e

echo "=== Data Pipeline Entrypoint ==="
echo "Creating log directories..."

# Create log directories (in case volumes aren't mounted properly)
mkdir -p /app/logs/ingested /app/logs/preprocessed /app/logs/validation

echo "Log directories ready:"
ls -la /app/logs/

echo "Running data pipeline with arguments: $@"

# Execute the main command
exec python src/data_pipeline/pipeline_data.py "$@"