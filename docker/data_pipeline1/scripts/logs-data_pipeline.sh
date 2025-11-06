#!/bin/bash
echo "Fetching Data Validation API logs ..."
CONTAINER_NAME="data_validation_api_alt"

if docker ps | grep -q $CONTAINER_NAME; then
    docker logs -f $CONTAINER_NAME
else
    docker logs $CONTAINER_NAME
fi