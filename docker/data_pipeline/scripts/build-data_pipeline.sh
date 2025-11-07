#!/bin/bash

# build-data_pipeline.sh
set -e

echo "Building Data Pipeline Docker image..."

# Navigate to project root
cd ../../..

# Build Data Pipeline image
echo "Building Data Pipeline image..."
docker build -t data-pipeline:latest -f docker/data_pipeline/Dockerfile .

echo "Build completed successfully!"
echo ""
echo "Available images:"
docker images | grep "data-pipeline"