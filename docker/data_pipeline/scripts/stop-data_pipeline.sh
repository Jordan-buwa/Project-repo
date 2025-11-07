#!/bin/bash

# stop-data_pipeline.sh
echo "Stopping Data Pipeline service..."

# Stop and remove container
docker stop data-pipeline 2>/dev/null || true
docker rm data-pipeline 2>/dev/null || true

echo "Data Pipeline stopped successfully!"