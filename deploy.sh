#!/bin/bash

echo "--- Starting Local Deployment ---"
set -e

echo "[1/2] Stopping and removing any old containers..."
docker compose down
echo "Old containers removed!"

echo "[2/2] Starting the application, Prometheus, and Grafana..."
docker compose up --build -d
echo "Deployment stack is starting in the background.."

echo ""
echo "--- Deployment Complete! ---"
echo "Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "Prometheus UI:     http://localhost:9090"
echo "API Health Check:  http://localhost:5001/health"
echo "API Docs:          http://localhost:5001/"
echo ""
echo "Useful command: 'docker compose logs -f' to view logs."