#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Load Environment Variables ---
# Assumes the .env file is in the project's root directory (one level up from this script)
ENV_FILE="$(dirname "$0")/../.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat "$ENV_FILE" | sed 's/#.*//g' | xargs)
else
    echo "Error: .env file not found in the project root ($ENV_FILE). Aborting."
    exit 1
fi

# --- Variables (Loaded from .env) ---
IMAGE_NAME="mlops-iris-classifier"
CONTAINER_NAME="iris-app"
MLFLOW_UI_PORT=${MLFLOW_UI_PORT:-5000} # Use default if not set
APP_PORT=${APP_PORT:-8000} # Use default if not set
DVC_REMOTE_DIR="/var/dvc-storage"

# Check for required variables
if [ -z "$DOCKERHUB_USERNAME" ]; then
    echo "Error: DOCKERHUB_USERNAME not set in .env file."
    exit 1
fi

echo "--- Starting Deployment for user: $DOCKERHUB_USERNAME ---"

# --- Setup DVC if not already initialized ---
if [ ! -d ".dvc" ]; then
    echo "DVC not initialized. Setting up DVC..."
    pip3 install --quiet dvc # Ensure dvc is installed
    dvc init --no-scm
    
    echo "Creating DVC local remote storage at $DVC_REMOTE_DIR"
    mkdir -p $DVC_REMOTE_DIR
    dvc remote add -d localremote $DVC_REMOTE_DIR
else
    echo "DVC already initialized."
fi

# --- Pull DVC data ---
echo "Pulling data from DVC remote..."
dvc pull -f

# --- Pull latest Docker image ---
echo "1. Pulling latest Docker image: $DOCKERHUB_USERNAME/$IMAGE_NAME:latest"
docker pull $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

# --- Stop and remove existing containers ---
echo "2. Stopping and removing existing containers..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true
docker stop mlflow-ui || true
docker rm mlflow-ui || true

# --- Run the new application container ---
echo "3. Running new application container"
docker run -d \
    --name $CONTAINER_NAME \
    -p $APP_PORT:$APP_PORT \
    --env-file $ENV_FILE \
    -v $(pwd)/mlruns:/app/mlruns \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

# --- Start MLflow UI ---
echo "4. Starting MLflow UI container"
docker run -d --name mlflow-ui \
    -p $MLFLOW_UI_PORT:$MLFLOW_UI_PORT \
    -v $(pwd)/mlruns:/mlruns \
    ghcr.io/mlflow/mlflow:v2.3.2 \
    mlflow ui --host 0.0.0.0 --port $MLFLOW_UI_PORT --backend-store-uri /mlruns

echo "--- Deployment Finished Successfully ---"
echo "Application is running on port $APP_PORT"
echo "MLflow UI is running on port $MLFLOW_UI_PORT"

