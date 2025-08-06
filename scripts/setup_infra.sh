#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Load Environment Variables ---
ENV_FILE="$(dirname "$0")/../.env"

# Use a more robust method to load environment variables that handles spaces and special characters.
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found. Using default values."
fi

echo "--- Ensuring a clean state by stopping existing services ---"
# Stop and remove containers, networks, and volumes created by 'up' in previous runs.
# This prevents potential conflicts and errors like 'ContainerConfig' mismatch.
docker-compose down --remove-orphans

echo "--- Starting MLOps Infrastructure ---"

# Create necessary directories if they don't exist. The -p flag prevents errors if they already exist.
mkdir -p ./minio_data
mkdir -p ./config
mkdir -p ./gitlab/config
mkdir -p ./gitlab/logs
mkdir -p ./gitlab/data


# Create Prometheus config only if it doesn't exist to avoid overwriting user changes.
if [ ! -f "./config/prometheus.yml" ]; then
    echo "Creating default prometheus.yml..."
    # Note: In a real scenario, you'd get the app's IP dynamically.
    # For now, we assume the app will be on the same Docker network.
    cat <<EOF > ./config/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'iris-app'
    static_configs:
      - targets: ['iris-app:8000']
EOF
fi

# Create Grafana config only if it doesn't exist.
if [ ! -f "./config/grafana-datasources.yml" ]; then
    echo "Creating default grafana-datasources.yml..."
    cat <<EOF > ./config/grafana-datasources.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
fi

# Start services using Docker Compose in detached mode.
# This command is idempotent: it only creates or updates containers that have changed.
docker-compose up -d

echo "--- Infrastructure is running ---"
echo "GitLab: http://localhost (The first run might take a few minutes to initialize)"
echo "MinIO UI: http://localhost:${MINIO_UI_PORT:-9001}"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000"
