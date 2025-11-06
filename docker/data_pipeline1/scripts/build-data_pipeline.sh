#!/bin/bash
echo "Building Data Validation API ..."
docker build -t data-validation-api-alt:latest -f ../Dockerfile .
echo "Alternative Data Validation API build completed!"