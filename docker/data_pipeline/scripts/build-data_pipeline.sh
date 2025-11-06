#!/bin/bash

# build-data_pipeline.sh
set -e

echo "Building Data Pipeline Docker images..."

# Build API Validation image
echo "Building API Validation image..."
docker build -t api-validation:latest -f Dockerfile.api .

# Build Data Pipeline image
echo "Building Data Pipeline image..."
docker build -t data-pipeline:latest -f Dockerfile.data .

echo "Build completed successfully!"
echo ""
echo "Available images:"
docker images | grep -E "(api-validation|data-pipeline)"