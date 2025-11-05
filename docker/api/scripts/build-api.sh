#!/bin/bash

echo "Building Churn Prediction API Docker image..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Build the API image with build args
docker build \
    -t churn-api-jaw:latest \
    -f docker/api/Dockerfile \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VERSION=1.0.0 \
    .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "API image built successfully!"
    echo ""
    echo "Available images:"
    docker images | grep churn-api-jaw
    echo ""
    echo "Next steps:"
    echo "  ./docker/api/scripts/start-api.sh  # Start the API"
else
    echo "API image build failed!"
    exit 1
fi