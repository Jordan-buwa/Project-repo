#!/bin/bash

set -e

echo "Starting Data Pipeline services..."

# Navigate to project root
cd ../../..

# Create network if it doesn't exist
docker network create data-pipeline-network 2>/dev/null || true

# Start API Validation service
echo "Starting API Validation service..."
docker run -d \
    --name api-validation \
    --network data-pipeline-network \
    -p 8000:8000 \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    --env-file .env \
    api-validation:latest

# Start Data Pipeline service
echo "Starting Data Pipeline service..."
docker run -d \
    --name data-pipeline \
    --network data-pipeline-network \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    --env-file .env \
    data-pipeline:latest

echo "Services started successfully!"
echo ""
echo "Running containers:"
docker ps --filter "name=api-validation\|data-pipeline"
echo ""
echo "API Validation available at: http://localhost:8000"