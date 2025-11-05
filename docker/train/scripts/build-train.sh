#!/bin/bash

echo "Building Churn Training Docker image..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Build the training image with build args
docker build \
    -t churn-training:latest \
    -f docker/train/Dockerfile \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VERSION=1.0.0 \
    .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Training image built successfully!"
    echo ""
    echo "Available images:"
    docker images | grep churn-training
    echo ""
    echo "Next steps:"
    echo "  ./docker/train/scripts/start-train.sh  # Start the training container"
else
    echo "Training image build failed!"
    exit 1
fi
