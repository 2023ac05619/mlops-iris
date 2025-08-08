#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Load Environment Variables ---
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found. Using default values."
fi

echo "--- Ensuring a clean state by stopping existing services ---"
docker-compose down --remove-orphans

echo "--- Starting MLOps Infrastructure ---"

# mkdir -p ./config

# Create Prometheus config 
if [ ! -f "./prometheus.yml" ]; then
    echo "Creating default prometheus.yml..."
    cat <<EOF > ./prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'iris-app'
    static_configs:
      - targets: ['app:8000'] # Assumes service name 'app' in docker-compose
EOF
fi

# Create Grafana config 
if [ ! -f "./grafana-datasources.yml" ]; then
    echo "Creating default grafana-datasources.yml..."
    cat <<EOF > ./grafana-datasources.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
fi

# Start services
docker-compose up -d

echo "--- Infrastructure is running ---"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000"