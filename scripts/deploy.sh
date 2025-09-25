#!/bin/bash

# Opinion Market Production Deployment Script
# This script handles the complete deployment process

set -e  # Exit on any error

# Configuration
APP_NAME="opinion-market"
NAMESPACE="opinion-market"
REGISTRY="ghcr.io"
IMAGE_TAG=${1:-latest}
ENVIRONMENT=${2:-production}

echo "🚀 Starting deployment of Opinion Market API"
echo "Environment: $ENVIRONMENT"
echo "Image Tag: $IMAGE_TAG"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        print_warning "Helm is not installed. Some features may not be available."
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Build the image
    docker build -f Dockerfile.production -t $REGISTRY/$APP_NAME:$IMAGE_TAG .
    
    # Push the image
    docker push $REGISTRY/$APP_NAME:$IMAGE_TAG
    
    print_success "Docker image built and pushed successfully"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    print_status "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Update image tag in deployment
    sed -i "s|image: .*|image: $REGISTRY/$APP_NAME:$IMAGE_TAG|g" deployment/kubernetes/opinion-market-deployment.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f deployment/kubernetes/ -n $NAMESPACE
    
    # Wait for deployment to be ready
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/opinion-market-api -n $NAMESPACE
    
    print_success "Deployment completed successfully"
}

# Run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Get the service URL
    SERVICE_URL=$(kubectl get service opinion-market-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_URL" ]; then
        # If no load balancer, use port-forward for testing
        print_warning "No load balancer found, using port-forward for health checks"
        kubectl port-forward service/opinion-market-service 8080:80 -n $NAMESPACE &
        PORT_FORWARD_PID=$!
        sleep 5
        SERVICE_URL="localhost:8080"
    fi
    
    # Health check endpoint
    if curl -f http://$SERVICE_URL/health; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        exit 1
    fi
    
    # Clean up port-forward if used
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID
    fi
}

# Run smoke tests
run_smoke_tests() {
    print_status "Running smoke tests..."
    
    # Get the service URL
    SERVICE_URL=$(kubectl get service opinion-market-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_URL" ]; then
        kubectl port-forward service/opinion-market-service 8080:80 -n $NAMESPACE &
        PORT_FORWARD_PID=$!
        sleep 5
        SERVICE_URL="localhost:8080"
    fi
    
    # Run basic API tests
    python scripts/smoke_tests.py --base-url http://$SERVICE_URL
    
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID
    fi
    
    print_success "Smoke tests passed"
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Deploy Prometheus if not exists
    if ! kubectl get deployment prometheus -n monitoring &> /dev/null; then
        kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring
    fi
    
    # Deploy Grafana dashboards
    kubectl apply -f deployment/monitoring/ -n $NAMESPACE
    
    print_success "Monitoring setup completed"
}

# Rollback deployment
rollback_deployment() {
    print_warning "Rolling back deployment..."
    
    kubectl rollout undo deployment/opinion-market-api -n $NAMESPACE
    kubectl rollout status deployment/opinion-market-api -n $NAMESPACE
    
    print_success "Rollback completed"
}

# Main deployment function
main() {
    print_status "Starting Opinion Market deployment process..."
    
    # Check prerequisites
    check_prerequisites
    
    # Build and push image
    build_and_push_image
    
    # Deploy to Kubernetes
    deploy_to_kubernetes
    
    # Run health checks
    run_health_checks
    
    # Run smoke tests
    run_smoke_tests
    
    # Setup monitoring
    setup_monitoring
    
    print_success "🎉 Deployment completed successfully!"
    print_status "Application is now running in the $ENVIRONMENT environment"
    
    # Show useful information
    echo ""
    echo "📊 Useful commands:"
    echo "  View pods: kubectl get pods -n $NAMESPACE"
    echo "  View logs: kubectl logs -f deployment/opinion-market-api -n $NAMESPACE"
    echo "  View service: kubectl get service -n $NAMESPACE"
    echo "  Scale deployment: kubectl scale deployment opinion-market-api --replicas=3 -n $NAMESPACE"
    echo ""
}

# Handle script arguments
case "${1:-}" in
    "rollback")
        rollback_deployment
        ;;
    "health-check")
        run_health_checks
        ;;
    "smoke-tests")
        run_smoke_tests
        ;;
    *)
        main
        ;;
esac
