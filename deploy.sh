#!/bin/bash

set -e  # Exit on any error

# Configuration
APP_NAME="mlops-iris-classifier"
CONTAINER_NAME="${APP_NAME}-app"
MLFLOW_CONTAINER_NAME="${APP_NAME}-mlflow"
NETWORK_NAME="${APP_NAME}-network"
API_PORT=${API_PORT:-8000}
MLFLOW_PORT=${MLFLOW_PORT:-5000}
DOCKER_IMAGE=${DOCKER_IMAGE:-"ghcr.io/yourusername/mlops-iris-classifier:latest"}
LOG_FILE="/var/log/${APP_NAME}-deploy.log"
BACKUP_DIR="/opt/${APP_NAME}/backups"
DATA_DIR="/opt/${APP_NAME}/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOG_FILE
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a $LOG_FILE
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a $LOG_FILE
}

# Check if running as root or with sudo
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        warning "Running as root - this is not recommended for production"
    fi
    
    # Check if user can run docker
    if ! docker info > /dev/null 2>&1; then
        error "Cannot run docker commands. Please add user to docker group or run with sudo"
    fi
}

# Create necessary directories
setup_directories() {
    log "Setting up directories..."
    
    sudo mkdir -p $BACKUP_DIR
    sudo mkdir -p $DATA_DIR
    sudo mkdir -p /var/log
    sudo mkdir -p /opt/${APP_NAME}/mlruns
    
    # Set proper permissions
    sudo chown -R $USER:$USER /opt/${APP_NAME}
    
    success "Directories created successfully"
}

# Create Docker network if it doesn't exist
setup_network() {
    log "Setting up Docker network..."
    
    if ! docker network ls | grep -q $NETWORK_NAME; then
        docker network create $NETWORK_NAME
        success "Docker network '$NETWORK_NAME' created"
    else
        log "Docker network '$NETWORK_NAME' already exists"
    fi
}

# Backup existing data if containers are running
backup_existing_data() {
    log "Backing up existing data..."
    
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        # Create backup of database and models
        BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
        
        mkdir -p $BACKUP_PATH
        
        # Copy database
        docker cp $CONTAINER_NAME:/app/logs/predictions.db $BACKUP_PATH/predictions.db 2>/dev/null || true
        
        # Copy models
        docker cp $CONTAINER_NAME:/app/models $BACKUP_PATH/ 2>/dev/null || true
        
        # Copy MLflow data
        docker cp $CONTAINER_NAME:/app/mlruns $BACKUP_PATH/ 2>/dev/null || true
        
        success "Data backed up to $BACKUP_PATH"
    else
        log "No existing container found, skipping backup"
    fi
}

# Stop and remove existing containers
cleanup_containers() {
    log "Cleaning up existing containers..."
    
    # Stop and remove main app container
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        log "Stopping existing app container..."
        docker stop $CONTAINER_NAME
    fi
    
    if docker ps -aq -f name=$CONTAINER_NAME | grep -q .; then
        log "Removing existing app container..."
        docker rm $CONTAINER_NAME
    fi
    
    # Stop and remove MLflow container
    if docker ps -q -f name=$MLFLOW_CONTAINER_NAME | grep -q .; then
        log "Stopping existing MLflow container..."
        docker stop $MLFLOW_CONTAINER_NAME
    fi
    
    if docker ps -aq -f name=$MLFLOW_CONTAINER_NAME | grep -q .; then
        log "Removing existing MLflow container..."
        docker rm $MLFLOW_CONTAINER_NAME
    fi
    
    success "Containers cleaned up"
}

# Pull latest Docker image
pull_image() {
    log "Pulling Docker image: $DOCKER_IMAGE"
    
    if ! docker pull $DOCKER_IMAGE; then
        error "Failed to pull Docker image: $DOCKER_IMAGE"
    fi
    
    success "Docker image pulled successfully"
}

# Start MLflow UI container
start_mlflow_ui() {
    log "Starting MLflow UI container..."
    
    docker run -d \
        --name $MLFLOW_CONTAINER_NAME \
        --network $NETWORK_NAME \
        -p $MLFLOW_PORT:5000 \
        -v /opt/${APP_NAME}/mlruns:/mlflow/mlruns \
        -e MLFLOW_BACKEND_STORE_URI=file:///mlflow/mlruns \
        --restart unless-stopped \
        python:3.9-slim \
        bash -c "pip install mlflow==2.5.0 && mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:///mlflow/mlruns"
    
    success "MLflow UI container started on port $MLFLOW_PORT"
}

# Start main application container
start_app_container() {
    log "Starting main application container..."
    
    docker run -d \
        --name $CONTAINER_NAME \
        --network $NETWORK_NAME \
        -p $API_PORT:8000 \
        -v /opt/${APP_NAME}/mlruns:/app/mlruns \
        -v /opt/${APP_NAME}/data:/app/data \
        -v $BACKUP_DIR:/app/backups \
        -e PYTHONUNBUFFERED=1 \
        -e MLFLOW_TRACKING_URI=http://${MLFLOW_CONTAINER_NAME}:5000 \
        -e GITHUB_TOKEN=${GITHUB_TOKEN:-""} \
        -e GITHUB_REPOSITORY=${GITHUB_REPOSITORY:-""} \
        --restart unless-stopped \
        --health-cmd="curl -f http://localhost:8000/health || exit 1" \
        --health-interval=30s \
        --health-timeout=10s \
        --health-retries=3 \
        $DOCKER_IMAGE
    
    success "Application container started on port $API_PORT"
}

# Wait for services to be healthy
wait_for_services() {
    log "Waiting for services to become healthy..."
    
    # Wait for MLflow UI
    for i in {1..30}; do
        if curl -s -f http://localhost:$MLFLOW_PORT > /dev/null; then
            success "MLflow UI is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            error "MLflow UI failed to start"
        fi
        sleep 2
    done
    
    # Wait for main application
    for i in {1..60}; do
        if curl -s -f http://localhost:$API_PORT/health > /dev/null; then
            success "Application is healthy"
            break
        fi
        if [ $i -eq 60 ]; then
            error "Application failed to start"
        fi
        sleep 2
    done
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    cat << EOF | sudo tee /etc/logrotate.d/${APP_NAME} > /dev/null
/var/log/${APP_NAME}-*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
    
    success "Log rotation configured"
}

# Setup monitoring and alerting
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create a simple monitoring script
    cat << 'EOF' | sudo tee /opt/${APP_NAME}/monitor.sh > /dev/null
#!/bin/bash
# Simple monitoring script for MLOps application

APP_NAME="mlops-iris-classifier"
CONTAINER_NAME="${APP_NAME}-app"
MLFLOW_CONTAINER_NAME="${APP_NAME}-mlflow"
API_PORT=8000
MLFLOW_PORT=5000

check_container() {
    local container=$1
    local service_name=$2
    
    if ! docker ps --format 'table {{.Names}}' | grep -q "^${container}$"; then
        echo "CRITICAL: $service_name container is not running"
        return 1
    fi
    
    return 0
}

check_api_health() {
    if ! curl -s -f http://localhost:$API_PORT/health > /dev/null; then
        echo "CRITICAL: API health check failed"
        return 1
    fi
    
    return 0
}

check_mlflow_ui() {
    if ! curl -s -f http://localhost:$MLFLOW_PORT > /dev/null; then
        echo "WARNING: MLflow UI is not responding"
        return 1
    fi
    
    return 0
}

# Main monitoring logic
main() {
    echo "$(date): Starting health check..."
    
    local issues=0
    
    check_container $CONTAINER_NAME "Application" || issues=$((issues + 1))
    check_container $MLFLOW_CONTAINER_NAME "MLflow UI" || issues=$((issues + 1))
    check_api_health || issues=$((issues + 1))
    check_mlflow_ui || issues=$((issues + 1))
    
    if [ $issues -eq 0 ]; then
        echo "$(date): All services are healthy"
    else
        echo "$(date): Found $issues issues"
    fi
    
    return $issues
}

main "$@"
EOF
    
    sudo chmod +x /opt/${APP_NAME}/monitor.sh
    
    # Add monitoring to crontab
    (crontab -l 2>/dev/null; echo "*/5 * * * * /opt/${APP_NAME}/monitor.sh >> /var/log/${APP_NAME}-monitor.log 2>&1") | crontab -
    
    success "Monitoring script configured"
}

# Display deployment information
show_deployment_info() {
    log "Deployment completed successfully!"
    
    echo ""
    echo "======================================"
    echo "    MLOps Iris Classifier Deployed"
    echo "======================================"
    echo ""
    echo " Application API: http://localhost:$API_PORT"
    echo " MLflow UI: http://localhost:$MLFLOW_PORT"
    echo " API Documentation: http://localhost:$API_PORT/docs"
    echo " Metrics: http://localhost:$API_PORT/metrics"
    echo " Data Directory: $DATA_DIR"
    echo " Backup Directory: $BACKUP_DIR"
    echo " Logs: $LOG_FILE"
    echo ""
    echo "Container Status:"
    docker ps --filter name=$APP_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "To monitor the application:"
    echo "  docker logs -f $CONTAINER_NAME"
    echo ""
    echo "To check system status:"
    echo "  curl http://localhost:$API_PORT/system_metrics"
    echo ""
}

# Cleanup function for failed deployments
cleanup_on_failure() {
    error "Deployment failed. Cleaning up..."
    
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    docker stop $MLFLOW_CONTAINER_NAME 2>/dev/null || true
    docker rm $MLFLOW_CONTAINER_NAME 2>/dev/null || true
    
    exit 1
}

# Main deployment function
main() {
    log "Starting MLOps Iris Classifier deployment..."
    
    # Set up error handling
    trap cleanup_on_failure ERR
    
    # Pre-deployment checks
    check_permissions
    
    # Setup
    setup_directories
    setup_network
    
    # Backup and cleanup
    backup_existing_data
    cleanup_containers
    
    # Deploy new version
    pull_image
    start_mlflow_ui
    start_app_container
    
    # Post-deployment setup
    wait_for_services
    setup_log_rotation
    setup_monitoring
    
    # Show results
    show_deployment_info
    
    success "Deployment completed successfully!"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi