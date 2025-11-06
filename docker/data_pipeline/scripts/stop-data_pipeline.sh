#!/bin/bash

# Stopping script for data pipeline
echo "Stopping Data Pipeline..."

CONTAINER_NAME="data_pipeline_container"

# Stopping the container
if docker stop $CONTAINER_NAME 2>/dev/null; then
    echo "Container $CONTAINER_NAME stopped successfully."
    
    # Removing the container
    docker rm $CONTAINER_NAME 2>/dev/null && echo "Container $CONTAINER_NAME removed."
else
    echo "Container $CONTAINER_NAME is not running or doesn't exist."
    
    # Checking if container exists but is stopped
    if docker ps -a | grep -q $CONTAINER_NAME; then
        echo "Removing stopped container..."
        docker rm $CONTAINER_NAME
    fi
fi

echo "Cleanup completed."