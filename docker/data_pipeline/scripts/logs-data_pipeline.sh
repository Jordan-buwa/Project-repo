#!/bin/bash

# Function to show usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -f           - Follow logs"
    echo "  -n LINES     - Number of lines to show"
    echo "  --since TIME - Show logs since timestamp"
    echo "  -h, --help   - Show this help message"
    exit 1
}

# Default values
FOLLOW=""
LINES=""
SINCE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

echo "=== Data Pipeline Logs ==="
docker logs data-pipeline $FOLLOW $LINES $SINCE