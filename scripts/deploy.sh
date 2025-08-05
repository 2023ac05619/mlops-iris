#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Load Environment Variables ---
ENV_FILE="$(dirname "$0")/../.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat "$ENV_FILE" | sed 's/#.*//g' | xargs)
else
    echo "Error: .env file not found. Aborting."
    exit 1
fi

# --- Variables ---
IMAGE_NAME="mlops-iris-classifier"
CONTAINER_NAME="iris-app"
MLFLOW_UI_PORT=${MLFLOW_UI_PORT:-5000}
APP_PORT=${APP_PORT:-8000}

# Check for required variables
if [ -z "$DOCKERHUB_USERNAME" ]; then
    echo "Error: DOCKERHUB_USERNAME not set in .env file."
    exit 1
fi

echo "--- Starting App Deployment ---"

# --- Setup DVC for MinIO ---
pip3 install --quiet "dvc[s3]"
if [ ! -d ".dvc" ]; then
    dvc init --no-scm
fi
dvc remote add --force origin s3://$DVC_REMOTE_BUCKET
dvc remote modify origin endpointurl $MINIO_ENDPOINT_URL
dvc remote modify origin access_key_id $MINIO_ACCESS_KEY
dvc remote modify origin secret_access_key $MINIO_SECRET_KEY

# --- Pull DVC data ---
echo "Pulling data from MinIO DVC remote..."
dvc pull -f

# --- Pull latest Docker image ---
echo "1. Pulling latest Docker image: $DOCKERHUB_USERNAME/$IMAGE_NAME:latest"
docker pull $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

# --- Stop and remove existing containers ---
echo "2. Stopping and removing existing app containers..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true
docker stop mlflow-ui || true
docker rm mlflow-ui || true

# --- Run the new application container ---
echo "3. Running new application container"
# Connect the app to the same network as the infrastructure
docker run -d \
    --name $CONTAINER_NAME \
    --network mlops-net \
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

echo "--- App Deployment Finished ---"