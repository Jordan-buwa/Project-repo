#!/bin/bash

# start-data_pipeline.sh
set -e

echo "Starting Data Pipeline service..."

# Navigate to project root
cd ../../..

# Create network if it doesn't exist
docker network create data-pipeline-network 2>/dev/null || true

# Start Data Pipeline service
echo "Starting Data Pipeline service..."
docker run -d \
    --name data-pipeline \
    --network data-pipeline-network \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/src/data_pipeline:/app/src/data_pipeline \
    --env-file .env \
    data-pipeline:latest

echo "Data Pipeline started successfully!"
echo ""
echo "Running containers:"
docker ps --filter "name=data-pipeline"
