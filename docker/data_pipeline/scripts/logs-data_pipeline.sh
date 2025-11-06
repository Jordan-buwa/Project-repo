#!/bin/bash

# Logs script for data pipeline
echo "Fetching Data Pipeline logs..."

CONTAINER_NAME="data_pipeline_container"

# Checking if container exists and is running
if docker ps | grep -q $CONTAINER_NAME; then
    echo "Showing logs for $CONTAINER_NAME:"
    docker logs -f $CONTAINER_NAME
elif docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Container exists but is not running. Showing last logs:"
    docker logs $CONTAINER_NAME
else
    echo "Container $CONTAINER_NAME not found."
    echo "Available containers:"
    docker ps -a
fi