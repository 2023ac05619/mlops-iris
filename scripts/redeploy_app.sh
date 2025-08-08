#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Load Environment Variables from .env file ---
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found at ${ENV_FILE}. Relying on pre-exported environment variables."
fi

# --- Configuration ---
IMAGE_TAG=${1}
# Check if an image tag was provided.
if [ -z "${IMAGE_TAG}" ]; then
    echo "Error: No image tag provided."
    echo "Usage: ./scripts/deploy.sh <tag>"
    exit 1
fi

# Check if Docker Hub username is set
if [ -z "${DOCKERHUB_USERNAME}" ]; then
    echo "Error: DOCKERHUB_USERNAME is not set. Please add it to your .env file or export it."
    exit 1
fi

FULL_IMAGE_NAME="${DOCKERHUB_REPO}:${IMAGE_TAG}"

echo "--- Starting Deployment ---"
echo "Image to deploy: ${FULL_IMAGE_NAME}"
sleep 1

# --- Docker Login ---
if [ -z "${DOCKERHUB_TOKEN}" ]; then
    echo "Error: DOCKERHUB_TOKEN is not set. Please add it to your .env file or export it."
    exit 1
fi
echo "Logging in to Docker Hub..."
echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin

# --- Pull the new image ---
echo "Pulling the latest Docker image..."
docker pull "${FULL_IMAGE_NAME}"

# --- Run Docker Compose ---
export IMAGE_TAG 

echo "Stopping and removing old containers..."
docker-compose down

echo "Starting new containers with image: ${FULL_IMAGE_NAME}"
docker-compose up -d --build
sleep 2
echo "--- Deployment Complete ---"
