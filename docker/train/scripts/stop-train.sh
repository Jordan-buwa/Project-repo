#!/bin/bash

echo "Stopping Training and MLflow containers..."

# Check if docker-compose file exists
if [ ! -f docker-compose.train.yml ]; then
    echo "docker-compose.train.yml not found!"
    exit 1
fi

# Stop and remove containers
docker-compose -f docker-compose.train.yml down

# Optional: clean up dangling images or volumes
if [ "$1" = "--clean" ]; then
    echo "Cleaning up dangling Docker resources..."
    docker system prune -f
    docker volume prune -f
fi

echo "Training environment stopped successfully."
