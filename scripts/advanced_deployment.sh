#!/bin/bash

# Advanced Deployment Script for Opinion Market Platform
# This script handles comprehensive deployment with health checks, rollbacks, and monitoring

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${1:-staging}"
VERSION="${2:-latest}"
NAMESPACE="opinion-market-${DEPLOYMENT_ENV}"
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
IMAGE_NAME="${REGISTRY}/opinion-market"
FULL_IMAGE="${IMAGE_NAME}:${VERSION}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        error_exit "kubectl is not installed or not in PATH"
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        error_exit "docker is not installed or not in PATH"
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed, some features may not work"
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build the image
    log_info "Building Docker image: $FULL_IMAGE"
    docker build -f Dockerfile.production -t "$FULL_IMAGE" . || error_exit "Docker build failed"
    
    # Push the image
    log_info "Pushing Docker image to registry..."
    docker push "$FULL_IMAGE" || error_exit "Docker push failed"
    
    log_success "Docker image built and pushed successfully"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes namespace: $NAMESPACE"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    
    # Update image in deployment manifests
    sed "s|IMAGE_PLACEHOLDER|$FULL_IMAGE|g" deployment/kubernetes/deployment.yaml | \
    kubectl apply -f - -n "$NAMESPACE"
    
    # Apply service
    kubectl apply -f deployment/kubernetes/service.yaml -n "$NAMESPACE"
    
    # Apply ingress
    kubectl apply -f deployment/kubernetes/ingress.yaml -n "$NAMESPACE"
    
    # Apply configmap
    kubectl apply -f deployment/kubernetes/configmap.yaml -n "$NAMESPACE"
    
    # Apply secrets
    if [ -f "deployment/kubernetes/secrets.yaml" ]; then
        kubectl apply -f deployment/kubernetes/secrets.yaml -n "$NAMESPACE"
    fi
    
    log_success "Kubernetes deployment completed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for deployment to be available
    kubectl wait --for=condition=available --timeout=300s deployment/opinion-market -n "$NAMESPACE" || \
    error_exit "Deployment failed to become ready within 5 minutes"
    
    # Wait for pods to be running
    kubectl wait --for=condition=ready --timeout=300s pod -l app=opinion-market -n "$NAMESPACE" || \
    error_exit "Pods failed to become ready within 5 minutes"
    
    log_success "Deployment is ready"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Get service URL
    SERVICE_URL=$(kubectl get service opinion-market -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$SERVICE_URL" ]; then
        SERVICE_URL=$(kubectl get service opinion-market -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    fi
    
    # Wait for service to be accessible
    log_info "Waiting for service to be accessible..."
    for i in {1..30}; do
        if curl -f "http://$SERVICE_URL:8000/health" &> /dev/null; then
            log_success "Service is accessible"
            break
        fi
        if [ $i -eq 30 ]; then
            error_exit "Service is not accessible after 5 minutes"
        fi
        sleep 10
    done
    
    # Run comprehensive health checks
    log_info "Running comprehensive health checks..."
    
    # Basic health check
    curl -f "http://$SERVICE_URL:8000/health" || error_exit "Basic health check failed"
    
    # Readiness check
    curl -f "http://$SERVICE_URL:8000/ready" || error_exit "Readiness check failed"
    
    # API health check
    curl -f "http://$SERVICE_URL:8000/api/v1/health" || error_exit "API health check failed"
    
    # Advanced features health checks
    log_info "Checking advanced features..."
    
    # Analytics engine health
    curl -f "http://$SERVICE_URL:8000/api/v1/advanced-analytics/health" || \
    log_warning "Analytics engine health check failed"
    
    # Auto-scaling health
    curl -f "http://$SERVICE_URL:8000/api/v1/auto-scaling/health" || \
    log_warning "Auto-scaling health check failed"
    
    # Alerting system health
    curl -f "http://$SERVICE_URL:8000/api/v1/intelligent-alerting/health" || \
    log_warning "Alerting system health check failed"
    
    log_success "Health checks completed"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get service URL
    SERVICE_URL=$(kubectl get service opinion-market -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$SERVICE_URL" ]; then
        SERVICE_URL=$(kubectl get service opinion-market -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    fi
    
    # Run smoke tests
    cd "$PROJECT_ROOT"
    python scripts/smoke_tests.py --base-url "http://$SERVICE_URL:8000" || \
    error_exit "Smoke tests failed"
    
    log_success "Smoke tests passed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Apply monitoring manifests
    if [ -f "deployment/monitoring/prometheus.yaml" ]; then
        kubectl apply -f deployment/monitoring/prometheus.yaml -n "$NAMESPACE"
    fi
    
    if [ -f "deployment/monitoring/grafana.yaml" ]; then
        kubectl apply -f deployment/monitoring/grafana.yaml -n "$NAMESPACE"
    fi
    
    # Apply service monitor
    if [ -f "deployment/monitoring/servicemonitor.yaml" ]; then
        kubectl apply -f deployment/monitoring/servicemonitor.yaml -n "$NAMESPACE"
    fi
    
    log_success "Monitoring setup completed"
}

# Setup alerting
setup_alerting() {
    log_info "Setting up alerting..."
    
    # Apply alerting rules
    if [ -f "deployment/monitoring/alertrules.yaml" ]; then
        kubectl apply -f deployment/monitoring/alertrules.yaml -n "$NAMESPACE"
    fi
    
    # Apply alertmanager config
    if [ -f "deployment/monitoring/alertmanager.yaml" ]; then
        kubectl apply -f deployment/monitoring/alertmanager.yaml -n "$NAMESPACE"
    fi
    
    log_success "Alerting setup completed"
}

# Backup current deployment
backup_deployment() {
    log_info "Creating backup of current deployment..."
    
    BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup current deployment
    kubectl get deployment opinion-market -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/deployment.yaml"
    kubectl get service opinion-market -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/service.yaml"
    kubectl get configmap opinion-market-config -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/configmap.yaml" 2>/dev/null || true
    
    log_success "Backup created at: $BACKUP_DIR"
    echo "$BACKUP_DIR" > "$PROJECT_ROOT/.last_backup"
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    if [ ! -f "$PROJECT_ROOT/.last_backup" ]; then
        error_exit "No backup found for rollback"
    fi
    
    BACKUP_DIR=$(cat "$PROJECT_ROOT/.last_backup")
    
    if [ ! -d "$BACKUP_DIR" ]; then
        error_exit "Backup directory not found: $BACKUP_DIR"
    fi
    
    # Restore deployment
    kubectl apply -f "$BACKUP_DIR/deployment.yaml" -n "$NAMESPACE"
    kubectl apply -f "$BACKUP_DIR/service.yaml" -n "$NAMESPACE"
    kubectl apply -f "$BACKUP_DIR/configmap.yaml" -n "$NAMESPACE" 2>/dev/null || true
    
    # Wait for rollback to complete
    kubectl rollout status deployment/opinion-market -n "$NAMESPACE" --timeout=300s
    
    log_success "Rollback completed"
}

# Cleanup old deployments
cleanup_old_deployments() {
    log_info "Cleaning up old deployments..."
    
    # Keep only last 5 deployments
    kubectl get replicasets -n "$NAMESPACE" -l app=opinion-market --sort-by=.metadata.creationTimestamp | \
    tail -n +6 | awk '{print $1}' | xargs -r kubectl delete replicaset -n "$NAMESPACE"
    
    log_success "Cleanup completed"
}

# Send notifications
send_notifications() {
    local status="$1"
    local message="$2"
    
    log_info "Sending notifications..."
    
    # Send Slack notification
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"Deployment $status: $message\"}" \
        "$SLACK_WEBHOOK_URL"
    fi
    
    # Send email notification
    if [ -n "${EMAIL_RECIPIENTS:-}" ]; then
        echo "Deployment $status: $message" | mail -s "Opinion Market Deployment $status" "$EMAIL_RECIPIENTS"
    fi
    
    log_success "Notifications sent"
}

# Main deployment function
main() {
    log_info "Starting deployment to $DEPLOYMENT_ENV environment with version $VERSION"
    
    # Check prerequisites
    check_prerequisites
    
    # Create backup
    backup_deployment
    
    # Build and push image
    build_and_push_image
    
    # Deploy to Kubernetes
    deploy_to_kubernetes
    
    # Wait for deployment
    wait_for_deployment
    
    # Run health checks
    run_health_checks
    
    # Run smoke tests
    run_smoke_tests
    
    # Setup monitoring
    setup_monitoring
    
    # Setup alerting
    setup_alerting
    
    # Cleanup old deployments
    cleanup_old_deployments
    
    # Send success notification
    send_notifications "SUCCESS" "Deployment to $DEPLOYMENT_ENV completed successfully"
    
    log_success "Deployment completed successfully!"
    
    # Display deployment information
    log_info "Deployment Information:"
    echo "  Environment: $DEPLOYMENT_ENV"
    echo "  Version: $VERSION"
    echo "  Namespace: $NAMESPACE"
    echo "  Image: $FULL_IMAGE"
    
    # Get service information
    SERVICE_URL=$(kubectl get service opinion-market -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "$SERVICE_URL" ]; then
        echo "  Service URL: http://$SERVICE_URL:8000"
    fi
}

# Handle script arguments
case "${1:-}" in
    "rollback")
        rollback_deployment
        send_notifications "ROLLBACK" "Deployment rolled back successfully"
        ;;
    "health-check")
        run_health_checks
        ;;
    "smoke-test")
        run_smoke_tests
        ;;
    *)
        main
        ;;
esac