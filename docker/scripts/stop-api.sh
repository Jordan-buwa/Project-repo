#!/bin/bash

echo "Stopping Churn Prediction API..."

docker-compose -f docker-compose.api.yml down

echo "API stopped"