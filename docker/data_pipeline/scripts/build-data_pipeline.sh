#!/bin/bash

# Build script for data pipeline
echo "Building Data Pipeline Docker image..."

# Set variables
IMAGE_NAME="data-pipeline"
VERSION="1.0.0"
DOCKERFILE="../Dockerfile"

# Building the Docker image
docker build -t $IMAGE_NAME:$VERSION -f $DOCKERFILE .

# Tagging as latest
docker tag $IMAGE_NAME:$VERSION $IMAGE_NAME:latest

echo "Build completed: $IMAGE_NAME:$VERSION"
echo "Available images:"
docker images | grep $IMAGE_NAME