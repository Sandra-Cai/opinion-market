#!/bin/bash

# Optimized Deployment Script for Opinion Market Platform
# Provides comprehensive deployment with health checks, rollback, and monitoring

set -euo pipefail

# Configuration
PROJECT_NAME="opinion-market"
VERSION=${1:-"latest"}
ENVIRONMENT=${2:-"production"}
NAMESPACE=${3:-"default"}
REGISTRY=${4:-"your-registry.com"}
IMAGE_TAG="${REGISTRY}/${PROJECT_NAME}:${VERSION}"

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
handle_error() {
    log_error "Deployment failed at line $1"
    log_info "Initiating rollback..."
    rollback_deployment
    exit 1
}

trap 'handle_error $LINENO' ERR

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if required tools are installed
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed. Aborting."; exit 1; }
    command -v docker >/dev/null 2>&1 || { log_error "docker is required but not installed. Aborting."; exit 1; }
    command -v helm >/dev/null 2>&1 || { log_error "helm is required but not installed. Aborting."; exit 1; }
    
    # Check kubectl connectivity
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster. Aborting."
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    log_success "Pre-deployment checks passed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building Docker image: $IMAGE_TAG"
    
    # Build image with build args
    docker build \
        --build-arg VERSION="$VERSION" \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        --tag "$IMAGE_TAG" \
        --file Dockerfile.robust \
        .
    
    # Push to registry
    log_info "Pushing image to registry..."
    docker push "$IMAGE_TAG"
    
    log_success "Image built and pushed successfully"
}

# Run security scan
security_scan() {
    log_info "Running security scan..."
    
    # Run Trivy security scan
    if command -v trivy >/dev/null 2>&1; then
        trivy image --exit-code 1 --severity HIGH,CRITICAL "$IMAGE_TAG"
        log_success "Security scan passed"
    else
        log_warning "Trivy not found, skipping security scan"
    fi
}

# Deploy with Helm
deploy_with_helm() {
    log_info "Deploying with Helm..."
    
    # Create values file for environment
    cat > "values-${ENVIRONMENT}.yaml" << EOF
image:
  repository: ${REGISTRY}/${PROJECT_NAME}
  tag: ${VERSION}
  pullPolicy: Always

environment: ${ENVIRONMENT}

resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: api.opinionmarket.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: opinion-market-tls
      hosts:
        - api.opinionmarket.com

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true

security:
  enabled: true
  networkPolicies:
    enabled: true
  podSecurityPolicy:
    enabled: true
EOF

    # Deploy with Helm
    helm upgrade --install "$PROJECT_NAME" \
        ./deployment/helm/opinion-market \
        --namespace "$NAMESPACE" \
        --values "values-${ENVIRONMENT}.yaml" \
        --wait \
        --timeout 10m \
        --create-namespace
    
    log_success "Helm deployment completed"
}

# Health checks
health_checks() {
    log_info "Running health checks..."
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available \
        --timeout=300s \
        deployment/"$PROJECT_NAME" \
        -n "$NAMESPACE"
    
    # Get pod status
    kubectl get pods -n "$NAMESPACE" -l app="$PROJECT_NAME"
    
    # Check service endpoints
    kubectl get endpoints -n "$NAMESPACE" -l app="$PROJECT_NAME"
    
    # Test application health endpoint
    local service_name="$PROJECT_NAME"
    local port="8000"
    
    # Port forward for health check
    kubectl port-forward -n "$NAMESPACE" "service/$service_name" "$port:$port" &
    local port_forward_pid=$!
    
    # Wait for port forward to be ready
    sleep 5
    
    # Health check
    if curl -f "http://localhost:$port/health" >/dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        kill $port_forward_pid 2>/dev/null || true
        return 1
    fi
    
    # Kill port forward
    kill $port_forward_pid 2>/dev/null || true
    
    # Performance check
    log_info "Running performance check..."
    if curl -f "http://localhost:$port/metrics" >/dev/null 2>&1; then
        log_success "Metrics endpoint accessible"
    else
        log_warning "Metrics endpoint not accessible"
    fi
    
    log_success "All health checks passed"
}

# Rollback function
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    # Get previous revision
    local previous_revision
    previous_revision=$(helm history "$PROJECT_NAME" -n "$NAMESPACE" --max 2 -o json | jq -r '.[0].revision')
    
    if [ "$previous_revision" != "null" ] && [ "$previous_revision" != "" ]; then
        helm rollback "$PROJECT_NAME" "$previous_revision" -n "$NAMESPACE"
        log_success "Rollback completed to revision $previous_revision"
    else
        log_warning "No previous revision found for rollback"
    fi
}

# Post-deployment tasks
post_deployment_tasks() {
    log_info "Running post-deployment tasks..."
    
    # Update monitoring dashboards
    if kubectl get configmap -n "$NAMESPACE" grafana-dashboards >/dev/null 2>&1; then
        log_info "Updating Grafana dashboards..."
        kubectl apply -f deployment/monitoring/grafana-dashboards/ -n "$NAMESPACE"
    fi
    
    # Update Prometheus rules
    if kubectl get prometheusrule -n "$NAMESPACE" opinion-market-rules >/dev/null 2>&1; then
        log_info "Updating Prometheus rules..."
        kubectl apply -f deployment/monitoring/prometheus-rules.yaml -n "$NAMESPACE"
    fi
    
    # Run security audit
    log_info "Running post-deployment security audit..."
    kubectl exec -n "$NAMESPACE" deployment/"$PROJECT_NAME" -- python -c "
from app.core.security_audit import security_auditor
import asyncio
result = asyncio.run(security_auditor.run_comprehensive_scan())
print(f'Security scan completed: {result}')
"
    
    log_success "Post-deployment tasks completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f "values-${ENVIRONMENT}.yaml"
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting optimized deployment for $PROJECT_NAME"
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Image: $IMAGE_TAG"
    
    # Run deployment steps
    pre_deployment_checks
    build_and_push_image
    security_scan
    deploy_with_helm
    health_checks
    post_deployment_tasks
    cleanup
    
    log_success "Deployment completed successfully!"
    
    # Display deployment information
    echo ""
    log_info "Deployment Information:"
    echo "  Project: $PROJECT_NAME"
    echo "  Version: $VERSION"
    echo "  Environment: $ENVIRONMENT"
    echo "  Namespace: $NAMESPACE"
    echo "  Image: $IMAGE_TAG"
    echo ""
    
    # Display useful commands
    log_info "Useful commands:"
    echo "  View pods: kubectl get pods -n $NAMESPACE"
    echo "  View logs: kubectl logs -f deployment/$PROJECT_NAME -n $NAMESPACE"
    echo "  Port forward: kubectl port-forward service/$PROJECT_NAME 8000:8000 -n $NAMESPACE"
    echo "  Health check: curl http://localhost:8000/health"
    echo "  Metrics: curl http://localhost:8000/metrics"
    echo ""
    
    log_success "Deployment completed successfully!"
}

# Show usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <version> [environment] [namespace] [registry]"
    echo ""
    echo "Examples:"
    echo "  $0 v1.2.3 production default your-registry.com"
    echo "  $0 latest staging staging-registry.com"
    echo "  $0 v1.0.0 development dev"
    echo ""
    echo "Default values:"
    echo "  version: latest"
    echo "  environment: production"
    echo "  namespace: default"
    echo "  registry: your-registry.com"
    exit 1
fi

# Run main function
main "$@"

