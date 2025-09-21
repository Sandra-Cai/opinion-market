#!/bin/bash

# Opinion Market Platform Deployment Script
# This script deploys the entire Opinion Market platform to production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
CLUSTER_NAME="opinion-market-${ENVIRONMENT}"
NAMESPACE="opinion-market"
REGION="us-east-1"
DOMAIN="opinionmarket.com"
BACKUP_ENABLED=${BACKUP_ENABLED:-true}
MONITORING_ENABLED=${MONITORING_ENABLED:-true}
SECURITY_SCAN_ENABLED=${SECURITY_SCAN_ENABLED:-true}
ROLLBACK_ENABLED=${ROLLBACK_ENABLED:-true}

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if required tools are installed
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
    command -v helm >/dev/null 2>&1 || error "helm is required but not installed"
    command -v terraform >/dev/null 2>&1 || error "terraform is required but not installed"
    command -v aws >/dev/null 2>&1 || error "aws CLI is required but not installed"
    command -v docker >/dev/null 2>&1 || error "docker is required but not installed"
    
    # Check optional tools
    command -v trivy >/dev/null 2>&1 && info "Trivy security scanner found" || warn "Trivy not found - security scanning disabled"
    command -v kube-score >/dev/null 2>&1 && info "kube-score found" || warn "kube-score not found - Kubernetes best practices check disabled"
    
    # Check AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        error "AWS credentials not configured"
    fi
    
    # Check kubectl context
    if ! kubectl cluster-info >/dev/null 2>&1; then
        error "kubectl is not connected to a cluster"
    fi
    
    # Check cluster resources
    check_cluster_resources
    
    log "Prerequisites check passed"
}

# Check cluster resources
check_cluster_resources() {
    log "Checking cluster resources..."
    
    # Check available nodes
    NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
    if [ $NODE_COUNT -lt 2 ]; then
        warn "Cluster has only $NODE_COUNT nodes. Consider using at least 2 nodes for production."
    fi
    
    # Check available memory
    AVAILABLE_MEMORY=$(kubectl top nodes --no-headers | awk '{sum+=$5} END {print sum}')
    if [ $AVAILABLE_MEMORY -lt 4000 ]; then
        warn "Low available memory: ${AVAILABLE_MEMORY}Mi. Consider scaling up nodes."
    fi
    
    # Check available CPU
    AVAILABLE_CPU=$(kubectl top nodes --no-headers | awk '{sum+=$3} END {print sum}')
    if [ $AVAILABLE_CPU -lt 2000 ]; then
        warn "Low available CPU: ${AVAILABLE_CPU}m. Consider scaling up nodes."
    fi
    
    info "Cluster resources check completed"
}

# Create deployment backup
create_backup() {
    if [ "$BACKUP_ENABLED" = "true" ]; then
        log "Creating deployment backup..."
        
        BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p $BACKUP_DIR
        
        # Backup current deployments
        kubectl get deployments -n $NAMESPACE -o yaml > $BACKUP_DIR/deployments.yaml
        kubectl get services -n $NAMESPACE -o yaml > $BACKUP_DIR/services.yaml
        kubectl get configmaps -n $NAMESPACE -o yaml > $BACKUP_DIR/configmaps.yaml
        kubectl get secrets -n $NAMESPACE -o yaml > $BACKUP_DIR/secrets.yaml
        
        # Backup database if exists
        if kubectl get deployment postgres -n $NAMESPACE >/dev/null 2>&1; then
            log "Creating database backup..."
            kubectl exec -n $NAMESPACE deployment/postgres -- pg_dump -U postgres opinion_market > $BACKUP_DIR/database.sql
        fi
        
        # Create backup metadata
        cat > $BACKUP_DIR/metadata.json << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "namespace": "$NAMESPACE",
    "cluster": "$CLUSTER_NAME",
    "backup_type": "pre_deployment"
}
EOF
        
        log "Backup created in $BACKUP_DIR"
        echo $BACKUP_DIR
    else
        info "Backup creation disabled"
        echo ""
    fi
}

# Run security scans
run_security_scans() {
    if [ "$SECURITY_SCAN_ENABLED" = "true" ]; then
        log "Running security scans..."
        
        # Scan Docker images
        if command -v trivy >/dev/null 2>&1; then
            log "Scanning Docker images for vulnerabilities..."
            trivy image --exit-code 1 --severity HIGH,CRITICAL ${REGISTRY}/api:latest || warn "Security vulnerabilities found in API image"
            trivy image --exit-code 1 --severity HIGH,CRITICAL ${REGISTRY}/worker:latest || warn "Security vulnerabilities found in worker image"
        fi
        
        # Scan Kubernetes manifests
        if command -v kube-score >/dev/null 2>&1; then
            log "Scanning Kubernetes manifests..."
            find deployment/kubernetes -name "*.yaml" -exec kube-score score {} \; || warn "Kubernetes best practices issues found"
        fi
        
        # Scan for secrets
        log "Scanning for exposed secrets..."
        kubectl get secrets -n $NAMESPACE -o yaml | grep -E "(password|secret|key|token)" && warn "Potential secrets found in cluster"
        
        log "Security scans completed"
    else
        info "Security scanning disabled"
    fi
}

# Health check after deployment
health_check() {
    log "Performing health check..."
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/api -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/worker -n $NAMESPACE
    
    # Check pod status
    FAILED_PODS=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase=Failed --no-headers | wc -l)
    if [ $FAILED_PODS -gt 0 ]; then
        error "$FAILED_PODS pods are in Failed state"
    fi
    
    # Check service endpoints
    kubectl get endpoints -n $NAMESPACE
    
    # Test API endpoint
    API_URL=$(kubectl get service api -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -n "$API_URL" ]; then
        log "Testing API endpoint: http://$API_URL/health"
        curl -f http://$API_URL/health || warn "API health check failed"
    fi
    
    log "Health check completed successfully"
}

# Rollback deployment
rollback_deployment() {
    if [ "$ROLLBACK_ENABLED" = "true" ]; then
        log "Rolling back deployment..."
        
        # Get last successful deployment
        LAST_DEPLOYMENT=$(kubectl rollout history deployment/api -n $NAMESPACE --no-headers | tail -1 | awk '{print $1}')
        
        if [ -n "$LAST_DEPLOYMENT" ]; then
            kubectl rollout undo deployment/api -n $NAMESPACE --to-revision=$LAST_DEPLOYMENT
            kubectl rollout undo deployment/worker -n $NAMESPACE --to-revision=$LAST_DEPLOYMENT
            
            # Wait for rollback to complete
            kubectl rollout status deployment/api -n $NAMESPACE
            kubectl rollout status deployment/worker -n $NAMESPACE
            
            log "Rollback completed successfully"
        else
            error "No previous deployment found for rollback"
        fi
    else
        info "Rollback disabled"
    fi
}

# Initialize Terraform
init_terraform() {
    log "Initializing Terraform..."
    
    cd deployment/terraform
    
    # Initialize Terraform
    terraform init
    
    # Plan the deployment
    terraform plan -var="environment=${ENVIRONMENT}" -out=tfplan
    
    # Ask for confirmation
    read -p "Do you want to apply the Terraform plan? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        terraform apply tfplan
    else
        warn "Terraform deployment cancelled"
        return 1
    fi
    
    cd ../..
}

# Build and push Docker images
build_images() {
    log "Building and pushing Docker images..."
    
    # Set Docker registry
    REGISTRY="ghcr.io/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\/[^/]*\).*/\1/')"
    
    # Build API image
    docker build -t ${REGISTRY}/api:latest -f Dockerfile .
    docker push ${REGISTRY}/api:latest
    
    # Build worker image
    docker build -t ${REGISTRY}/worker:latest -f Dockerfile.worker .
    docker push ${REGISTRY}/worker:latest
    
    log "Docker images built and pushed successfully"
}

# Deploy infrastructure components
deploy_infrastructure() {
    log "Deploying infrastructure components..."
    
    # Create namespace
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy network policies
    kubectl apply -f deployment/security/network-policies.yaml
    
    # Deploy monitoring stack
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace ${NAMESPACE} \
        --set prometheus.prometheusSpec.retention=30d \
        --set grafana.enabled=true \
        --set grafana.adminPassword=admin123 \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=10Gi
    
    # Deploy custom Prometheus rules
    kubectl apply -f deployment/monitoring/prometheus-rules.yaml
    
    # Deploy Grafana dashboards
    kubectl apply -f deployment/monitoring/grafana-dashboards/
    
    log "Infrastructure components deployed successfully"
}

# Deploy database and cache
deploy_databases() {
    log "Deploying databases..."
    
    # Deploy PostgreSQL
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    helm upgrade --install postgres bitnami/postgresql \
        --namespace ${NAMESPACE} \
        --set postgresqlPassword=password \
        --set postgresqlDatabase=opinion_market \
        --set persistence.enabled=true \
        --set persistence.size=10Gi \
        --set metrics.enabled=true \
        --set metrics.serviceMonitor.enabled=true
    
    # Deploy Redis
    helm upgrade --install redis bitnami/redis \
        --namespace ${NAMESPACE} \
        --set auth.enabled=false \
        --set persistence.enabled=true \
        --set persistence.size=5Gi \
        --set metrics.enabled=true \
        --set metrics.serviceMonitor.enabled=true \
        --set architecture=standalone
    
    # Wait for databases to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n ${NAMESPACE} --timeout=300s
    
    log "Databases deployed successfully"
}

# Deploy application
deploy_application() {
    log "Deploying application..."
    
    # Get database connection details
    POSTGRES_HOST=$(kubectl get svc postgres-postgresql -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
    REDIS_HOST=$(kubectl get svc redis-master -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
    
    # Deploy using Helm
    helm upgrade --install opinion-market deployment/helm/opinion-market \
        --namespace ${NAMESPACE} \
        --set app.image.tag=latest \
        --set app.config.databaseUrl="postgresql://opinion_market:password@${POSTGRES_HOST}:5432/opinion_market" \
        --set app.config.redisUrl="redis://${REDIS_HOST}:6379" \
        --set global.environment=${ENVIRONMENT} \
        --set app.ingress.enabled=true \
        --set app.ingress.hosts[0].host="api.${DOMAIN}" \
        --set app.ingress.tls[0].hosts[0]="api.${DOMAIN}"
    
    # Wait for application to be ready
    kubectl wait --for=condition=available deployment/opinion-market-api -n ${NAMESPACE} --timeout=300s
    
    log "Application deployed successfully"
}

# Deploy backup system
deploy_backup() {
    log "Deploying backup system..."
    
    # Deploy backup configuration
    kubectl apply -f deployment/backup/backup-config.yaml
    
    # Create backup secrets
    kubectl create secret generic backup-secrets \
        --namespace ${NAMESPACE} \
        --from-literal=AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
        --from-literal=AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log "Backup system deployed successfully"
}

# Configure monitoring
configure_monitoring() {
    log "Configuring monitoring..."
    
    # Deploy custom metrics
    kubectl apply -f deployment/monitoring/custom-metrics/
    
    # Configure alerting
    kubectl apply -f deployment/monitoring/alertmanager-config.yaml
    
    # Deploy service monitors
    kubectl apply -f deployment/monitoring/service-monitors/
    
    log "Monitoring configured successfully"
}

# Run health checks
health_checks() {
    log "Running health checks..."
    
    # Check if all pods are running
    if ! kubectl get pods -n ${NAMESPACE} --field-selector=status.phase!=Running | grep -q .; then
        log "All pods are running"
    else
        error "Some pods are not running"
    fi
    
    # Check API health
    API_URL=$(kubectl get ingress -n ${NAMESPACE} -o jsonpath='{.items[0].spec.rules[0].host}')
    if curl -f https://${API_URL}/health >/dev/null 2>&1; then
        log "API health check passed"
    else
        error "API health check failed"
    fi
    
    # Check database connectivity
    if kubectl exec -n ${NAMESPACE} deployment/opinion-market-api -- pg_isready -h postgres-postgresql -p 5432 >/dev/null 2>&1; then
        log "Database connectivity check passed"
    else
        error "Database connectivity check failed"
    fi
    
    # Check Redis connectivity
    if kubectl exec -n ${NAMESPACE} deployment/opinion-market-api -- redis-cli -h redis-master ping >/dev/null 2>&1; then
        log "Redis connectivity check passed"
    else
        error "Redis connectivity check failed"
    fi
    
    log "All health checks passed"
}

# Configure SSL certificates
configure_ssl() {
    log "Configuring SSL certificates..."
    
    # Install cert-manager
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
    
    # Wait for cert-manager to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=cert-manager -n cert-manager --timeout=300s
    
    # Configure Let's Encrypt cluster issuer
    kubectl apply -f deployment/ssl/cluster-issuer.yaml
    
    # Apply certificate
    kubectl apply -f deployment/ssl/certificate.yaml
    
    log "SSL certificates configured successfully"
}

# Configure load balancing
configure_load_balancing() {
    log "Configuring load balancing..."
    
    # Install NGINX ingress controller
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update
    
    helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --set controller.service.type=LoadBalancer \
        --set controller.ingressClassResource.name=nginx
    
    log "Load balancing configured successfully"
}

# Run performance tests
performance_tests() {
    log "Running performance tests..."
    
    # Install k6 for performance testing
    kubectl apply -f deployment/testing/k6-operator.yaml
    
    # Run performance tests
    kubectl apply -f deployment/testing/performance-tests.yaml
    
    # Wait for tests to complete
    kubectl wait --for=condition=complete job/performance-test -n ${NAMESPACE} --timeout=600s
    
    # Get test results
    kubectl logs job/performance-test -n ${NAMESPACE}
    
    log "Performance tests completed"
}

# Configure logging
configure_logging() {
    log "Configuring logging..."
    
    # Install Fluent Bit
    helm repo add fluent https://fluent.github.io/helm-charts
    helm repo update
    
    helm upgrade --install fluent-bit fluent/fluent-bit \
        --namespace ${NAMESPACE} \
        --set config.outputs=null \
        --set config.filters=null
    
    log "Logging configured successfully"
}

# Final verification
final_verification() {
    log "Performing final verification..."
    
    # Check all services
    kubectl get svc -n ${NAMESPACE}
    
    # Check all deployments
    kubectl get deployments -n ${NAMESPACE}
    
    # Check ingress
    kubectl get ingress -n ${NAMESPACE}
    
    # Check certificates
    kubectl get certificates -n ${NAMESPACE}
    
    # Check monitoring
    kubectl get servicemonitors -n ${NAMESPACE}
    
    log "Final verification completed"
}

# Main deployment function
main() {
    log "Starting Opinion Market platform deployment to ${ENVIRONMENT}"
    
    # Check prerequisites
    check_prerequisites
    
    # Create backup before deployment
    BACKUP_DIR=$(create_backup)
    
    # Run security scans
    run_security_scans
    
    # Initialize Terraform (optional - comment out if using existing infrastructure)
    # init_terraform
    
    # Build and push images
    build_images
    
    # Deploy infrastructure
    deploy_infrastructure
    
    # Deploy databases
    deploy_databases
    
    # Deploy application
    deploy_application
    
    # Deploy backup system
    deploy_backup
    
    # Configure monitoring
    configure_monitoring
    
    # Configure SSL certificates
    configure_ssl
    
    # Configure load balancing
    configure_load_balancing
    
    # Configure logging
    configure_logging
    
    # Run health checks
    health_check
    
    # Run performance tests
    performance_tests
    
    # Final verification
    final_verification
    
    log "Opinion Market platform deployment completed successfully!"
    log "Access your application at: https://api.${DOMAIN}"
    
    # Cleanup old backups (keep last 5)
    if [ -n "$BACKUP_DIR" ]; then
        log "Cleaning up old backups..."
        find backups -type d -name "backup_*" | sort | head -n -5 | xargs rm -rf
    fi
    log "Access Grafana at: https://grafana.${DOMAIN}"
    log "Access Prometheus at: https://prometheus.${DOMAIN}"
}

# Error handling with rollback
handle_deployment_error() {
    error "Deployment failed at step: $1"
    
    if [ "$ROLLBACK_ENABLED" = "true" ] && [ -n "$BACKUP_DIR" ]; then
        log "Attempting to rollback deployment..."
        rollback_deployment
    fi
    
    exit 1
}

# Run main function with error handling
set -e
trap 'handle_deployment_error "Unknown error"' ERR

main "$@"
