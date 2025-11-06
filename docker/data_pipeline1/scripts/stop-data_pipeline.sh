#!/bin/bash

# Alternative stop script for data_pipeline1
echo "Stopping Data Pipeline (Alternative Configuration)..."

CONTAINER_NAME="data_pipeline_alt_container"

# Graceful shutdown with timeout
echo "Initiating graceful shutdown..."
docker stop -t 30 $CONTAINER_NAME 2>/dev/null

if [ $? -eq 0 ]; then
    echo "Container $CONTAINER_NAME stopped gracefully."
    docker rm $CONTAINER_NAME 2>/dev/null && echo "Container removed."
else
    echo "Graceful shutdown failed or container not found."
    echo "Forcing removal..."
    docker rm -f $CONTAINER_NAME 2>/dev/null && echo "Container forced removed."
fi

echo "Data_pipeline1 cleanup completed."