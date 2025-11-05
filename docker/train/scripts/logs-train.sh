#!/bin/bash

echo "Showing Training and MLflow logs..."

# Check if services are running
if [ -z "$(docker-compose -f docker-compose.train.yml ps -q mlflow)" ]; then
    echo "Training environment is not running. Start it with: ./docker/train/scripts/start-train.sh"
    exit 1
fi

# Show logs with options
if [ "$1" = "-f" ]; then
    docker-compose -f docker-compose.train.yml logs -f
else
    docker-compose -f docker-compose.train.yml logs
fi
