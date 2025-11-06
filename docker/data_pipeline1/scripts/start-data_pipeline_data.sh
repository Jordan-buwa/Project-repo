#!/bin/bash
echo "Starting Data Validation API (Alternative)..."

docker stop data_validation_api_alt 2>/dev/null
docker rm data_validation_api_alt 2>/dev/null

docker run -d \
    --name data_validation_api_alt \
    -p 8081:8080 \
    -v $(pwd)/../models:/app/models \
    -v $(pwd)/../config:/app/config \
    -v $(pwd)/../data:/app/data \
    -v $(pwd)/../src:/app/src \
    data-validation-api-alt:latest

echo "Alternative Data Validation API started on port 8081"