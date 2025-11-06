#!/bin/bash

# stop-data_pipeline.sh
echo "Stopping Data Pipeline services..."

# Stop and remove containers
docker stop data-pipeline api-validation 2>/dev/null || true
docker rm data-pipeline api-validation 2>/dev/null || true

echo "Services stopped successfully!"