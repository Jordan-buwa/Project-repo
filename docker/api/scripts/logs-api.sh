#!/bin/bash

echo "Showing API logs..."

# Check if services are running
if [ -z "$(docker-compose -f docker-compose.api.yml ps -q churn-api-jaw)" ]; then
    echo "API is not running. Start it with: ./docker/api/scripts/start-api.sh"
    exit 1
fi

# Show logs with options
if [ "$1" = "-f" ]; then
    docker-compose -f docker-compose.api.yml logs -f
else
    docker-compose -f docker-compose.api.yml logs
fi