#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Load Environment Variables ---
ENV_FILE="$(dirname "$0")/../.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat "$ENV_FILE" | sed 's/#.*//g' | xargs)
else
    echo "Warning: .env file not found. Using default values."
fi

echo "--- Starting MLOps Infrastructure ---"

# Create necessary directories if they don't exist
mkdir -p ./minio_data
mkdir -p ./config

# Create Prometheus config if it doesn't exist
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

# Create Grafana config if it doesn't exist
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

# Start services using Docker Compose
docker-compose up -d

echo "--- Infrastructure is running ---"
echo "MinIO UI: http://localhost:${MINIO_UI_PORT:-9001}"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000"