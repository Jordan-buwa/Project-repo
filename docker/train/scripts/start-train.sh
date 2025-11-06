#!/bin/bash

echo "Starting Training Environment (MLflow + Training Container)..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo ".env file not found, creating from .env.example if exists..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Created .env file from .env.example"
    else
        echo "No .env or .env.example found. Please create one before proceeding."
        exit 1
    fi
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)
echo "Environment variables loaded."

# Create necessary directories
mkdir -p models artifacts logs mlruns data/raw data/processed

# Start MLflow service
echo "Starting MLflow container..."
docker-compose -f docker-compose.train.yml up -d mlflow

# Wait for MLflow to be ready
echo "⏳ Waiting for MLflow to start..."
for i in {1..20}; do
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        echo "MLflow is up and running!"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:5000 > /dev/null 2>&1; then
    echo "MLflow failed to start within 40 seconds."
    docker-compose -f docker-compose.train.yml logs mlflow
    exit 1
fi

echo ""
echo "MLflow UI: http://localhost:5000"
echo "Model Directory: ./models"
echo "Logs Directory: ./logs"
echo "Artifacts Directory: ./artifacts"
echo ""

# Array of training scripts
TRAIN_SCRIPTS=("train_rf.py" "train_xgb.py" "train_nn.py")

# Run each training script in sequence
for SCRIPT in "${TRAIN_SCRIPTS[@]}"; do
    echo "Starting training: $SCRIPT ..."
    
    docker run --rm \
        -v $(pwd)/mlruns:/app/mlruns \
        -v $(pwd)/artifacts:/app/artifacts \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/data:/app/data \
        churn-training:latest \
        bash -c "python src/models/$SCRIPT"

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Training completed successfully: $SCRIPT"
    else
        echo "Training failed: $SCRIPT. Continuing with next script..."
    fi
done

echo "All training scripts finished (some may have failed, check logs)."
echo "Use ./docker/train/scripts/logs-train.sh to view container logs."
