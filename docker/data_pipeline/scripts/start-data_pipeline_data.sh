#!/bin/bash

# Starting script for data pipeline
echo "Starting Data Pipeline..."

CONTAINER_NAME="data_pipeline_container"
IMAGE_NAME="data-pipeline:latest"
NETWORK_NAME="data-pipeline-network"

# Creating network if it doesn't exist
docker network create $NETWORK_NAME 2>/dev/null || true

# Stopping and remove existing container if running
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Running the container
docker run -d \
    --name $CONTAINER_NAME \
    --network $NETWORK_NAME \
    -p 8080:8080 \
    -v $(pwd)/../data:/app/data \
    -v $(pwd)/../logs:/app/logs \
    -e ENVIRONMENT=production \
    -e LOG_LEVEL=INFO \
    $IMAGE_NAME

echo "Data Pipeline started successfully!"
echo "Container name: $CONTAINER_NAME"
echo "Check logs with: ./logs-data_pipeline.sh"