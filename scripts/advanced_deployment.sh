#!/bin/bash

# Advanced Deployment Automation Script
# Comprehensive deployment with rollback, health checks, and monitoring

set -e

# Configuration
APP_NAME="opinion-market-api"
NAMESPACE="opinion-market"
IMAGE_TAG="${1:-latest}"
ENVIRONMENT="${2:-staging}"
REPLICAS="${3:-3}"
HEALTH_CHECK_TIMEOUT=300
ROLLBACK_TIMEOUT=60

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

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    log_success "Pre-deployment checks passed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Build image
    docker build -t "$APP_NAME:$IMAGE_TAG" -f Dockerfile.robust .
    
    # Tag for registry (assuming Docker Hub for this example)
    docker tag "$APP_NAME:$IMAGE_TAG" "your-registry/$APP_NAME:$IMAGE_TAG"
    
    # Push image
    docker push "your-registry/$APP_NAME:$IMAGE_TAG"
    
    log_success "Image built and pushed successfully"
}

# Deploy application
deploy_application() {
    log_info "Deploying application to $ENVIRONMENT environment..."
    
    # Create deployment YAML
    cat > deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $APP_NAME
  namespace: $NAMESPACE
  labels:
    app: $APP_NAME
    environment: $ENVIRONMENT
spec:
  replicas: $REPLICAS
  selector:
    matchLabels:
      app: $APP_NAME
  template:
    metadata:
      labels:
        app: $APP_NAME
        environment: $ENVIRONMENT
    spec:
      containers:
      - name: $APP_NAME
        image: your-registry/$APP_NAME:$IMAGE_TAG
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "$ENVIRONMENT"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: $APP_NAME-service
  namespace: $NAMESPACE
spec:
  selector:
    app: $APP_NAME
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: $APP_NAME-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $APP_NAME
  minReplicas: $REPLICAS
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
    
    # Apply deployment
    kubectl apply -f deployment.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=${HEALTH_CHECK_TIMEOUT}s deployment/$APP_NAME -n $NAMESPACE
    
    log_success "Application deployed successfully"
}

# Health check
health_check() {
    log_info "Running health checks..."
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service $APP_NAME-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_IP" ]; then
        log_warning "LoadBalancer IP not available, using port-forward"
        kubectl port-forward service/$APP_NAME-service 8000:80 -n $NAMESPACE &
        PORT_FORWARD_PID=$!
        sleep 5
        SERVICE_IP="localhost"
    fi
    
    # Health check endpoints
    ENDPOINTS=("/health" "/ready" "/metrics")
    
    for endpoint in "${ENDPOINTS[@]}"; do
        log_info "Checking endpoint: $endpoint"
        
        for i in {1..10}; do
            if curl -f -s "http://$SERVICE_IP:8000$endpoint" > /dev/null; then
                log_success "Endpoint $endpoint is healthy"
                break
            else
                if [ $i -eq 10 ]; then
                    log_error "Endpoint $endpoint failed health check"
                    return 1
                fi
                log_warning "Endpoint $endpoint not ready, retrying in 10s... (attempt $i/10)"
                sleep 10
            fi
        done
    done
    
    # Clean up port-forward if used
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
    
    log_success "All health checks passed"
}

# Performance test
performance_test() {
    log_info "Running performance tests..."
    
    # Install hey if not available
    if ! command -v hey &> /dev/null; then
        log_info "Installing hey for performance testing..."
        go install github.com/rakyll/hey@latest
    fi
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service $APP_NAME-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_IP" ]; then
        kubectl port-forward service/$APP_NAME-service 8000:80 -n $NAMESPACE &
        PORT_FORWARD_PID=$!
        sleep 5
        SERVICE_IP="localhost"
    fi
    
    # Run performance test
    hey -n 1000 -c 10 "http://$SERVICE_IP:8000/health" > performance_results.txt
    
    # Check performance results
    if grep -q "200 responses" performance_results.txt; then
        log_success "Performance test passed"
    else
        log_error "Performance test failed"
        return 1
    fi
    
    # Clean up
    rm -f performance_results.txt
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
}

# Rollback function
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Get previous deployment
    PREVIOUS_REVISION=$(kubectl rollout history deployment/$APP_NAME -n $NAMESPACE --no-headers | tail -2 | head -1 | awk '{print $1}')
    
    if [ -z "$PREVIOUS_REVISION" ]; then
        log_error "No previous revision found for rollback"
        return 1
    fi
    
    # Rollback
    kubectl rollout undo deployment/$APP_NAME -n $NAMESPACE --to-revision=$PREVIOUS_REVISION
    
    # Wait for rollback to complete
    kubectl wait --for=condition=available --timeout=${ROLLBACK_TIMEOUT}s deployment/$APP_NAME -n $NAMESPACE
    
    log_success "Rollback completed"
}

# Monitoring setup
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create monitoring namespace if it doesn't exist
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus (simplified)
    cat > prometheus-config.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'opinion-market-api'
      static_configs:
      - targets: ['$APP_NAME-service.$NAMESPACE.svc.cluster.local:80']
      metrics_path: '/metrics'
      scrape_interval: 5s
EOF
    
    kubectl apply -f prometheus-config.yaml
    
    log_success "Monitoring setup completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f deployment.yaml prometheus-config.yaml
}

# Main deployment function
main() {
    log_info "Starting advanced deployment for $APP_NAME"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image tag: $IMAGE_TAG"
    log_info "Replicas: $REPLICAS"
    
    # Set up error handling
    trap cleanup EXIT
    trap 'log_error "Deployment failed, initiating rollback..."; rollback_deployment; exit 1' ERR
    
    # Execute deployment steps
    pre_deployment_checks
    build_and_push_image
    deploy_application
    health_check
    performance_test
    setup_monitoring
    
    log_success "Deployment completed successfully!"
    log_info "Application is available at: http://$(kubectl get service $APP_NAME-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
    
    # Show deployment status
    kubectl get pods -n $NAMESPACE -l app=$APP_NAME
    kubectl get services -n $NAMESPACE
}

# Run main function
main "$@"
