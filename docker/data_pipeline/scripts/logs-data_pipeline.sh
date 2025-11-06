#!/bin/bash

# logs-data_pipeline.sh

# Function to show usage
usage() {
    echo "Usage: $0 [service] [options]"
    echo "Services:"
    echo "  api          - Show API Validation logs"
    echo "  data         - Show Data Pipeline logs"
    echo "  all          - Show all logs (default)"
    echo "Options:"
    echo "  -f           - Follow logs"
    echo "  -n LINES     - Number of lines to show"
    echo "  --since TIME - Show logs since timestamp"
    exit 1
}

# Default values
SERVICE="all"
FOLLOW=""
LINES=""
SINCE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        api|data|all)
            SERVICE=$1
            shift
            ;;
        -f)
            FOLLOW="-f"
            shift
            ;;
        -n)
            LINES="--tail=$2"
            shift 2
            ;;
        --since)
            SINCE="--since=$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Show logs based on service
case $SERVICE in
    api)
        echo "=== API Validation Logs ==="
        docker logs api-validation $FOLLOW $LINES $SINCE
        ;;
    data)
        echo "=== Data Pipeline Logs ==="
        docker logs data-pipeline $FOLLOW $LINES $SINCE
        ;;
    all)
        echo "=== API Validation Logs ==="
        docker logs api-validation $LINES $SINCE
        echo ""
        echo "=== Data Pipeline Logs ==="
        docker logs data-pipeline $LINES $SINCE
        ;;
esac