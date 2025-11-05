#!/bin/bash

echo "Starting Churn Prediction API..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo ".env file not found, creating from .env.example if exists..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Created .env file from .env.example"
    fi
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Loaded environment variables from .env"
fi


# Check if image exists, if not build it
if [[ "$(docker images -q churn-api-jaw:latest 2> /dev/null)" == "" ]]; then
    echo "Image not found, building..."
    ./docker/api/scripts/build-api.sh
fi

# Create necessary directories
mkdir -p models artifacts data/processed data/raw logs mlruns

# Start the API service
echo "Starting API container..."
docker-compose -f docker-compose.api.yml up -d

# Wait for services to be healthy
echo "Waiting for services to be ready..."

# Wait for API to be healthy
echo "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is healthy and running!"
        echo ""
        echo "API URLs:"
        echo "   - API: http://localhost:8000"
        echo "   - Docs: http://localhost:8000/docs"
        echo "   - Health: http://localhost:8000/health"
        echo ""
        echo "View logs: docker-compose -f docker-compose.api.yml logs -f"
        exit 0
    fi
    sleep 2
done

echo "API failed to start within 60 seconds"
echo "Check logs: ./docker/api/scripts/logs-api.sh"
docker-compose -f docker-compose.api.yml logs churn-api-jaw
exit 1