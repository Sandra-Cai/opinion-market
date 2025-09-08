#!/bin/bash

# 🚀 Automated Deployment & Rollback System
# Advanced deployment automation with health checks, rollback capabilities, and blue-green deployments

set -euo pipefail

# Configuration
DEPLOY_LOG="/tmp/deploy.log"
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
ROLLBACK_TIMEOUT=120      # 2 minutes
MAX_RETRIES=3
DEPLOYMENT_HISTORY="/tmp/deployment_history.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Deployment strategies
BLUE_GREEN="blue-green"
ROLLING="rolling"
CANARY="canary"

# Initialize deployment system
init_deployment() {
    echo -e "${PURPLE}🚀 Initializing Automated Deployment System${NC}"
    
    # Create deployment history if it doesn't exist
    if [[ ! -f "$DEPLOYMENT_HISTORY" ]]; then
        echo '{"deployments": [], "current_version": null, "rollback_version": null}' > "$DEPLOYMENT_HISTORY"
    fi
    
    # Set up signal handlers
    trap cleanup_deployment EXIT INT TERM
    
    echo -e "${GREEN}✅ Deployment system initialized${NC}"
}

# Logging functions
log_deploy() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$DEPLOY_LOG"
}

log_deploy_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$DEPLOY_LOG"
}

log_deploy_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$DEPLOY_LOG"
}

log_deploy_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$DEPLOY_LOG"
}

# Validate deployment prerequisites
validate_prerequisites() {
    local environment="$1"
    local version="$2"
    
    log_deploy "Validating deployment prerequisites for $environment..."
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_deploy_error "Docker is not installed or not in PATH"
        return 1
    fi
    
    # Check if kubectl is available (for Kubernetes deployments)
    if [[ "$environment" == "production" ]] && ! command -v kubectl &> /dev/null; then
        log_deploy_warning "kubectl not found, using Docker Compose for production"
    fi
    
    # Check if image exists
    if ! docker image inspect "opinion-market:$version" &> /dev/null; then
        log_deploy_error "Docker image opinion-market:$version not found"
        return 1
    fi
    
    # Check disk space
    local disk_usage=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        log_deploy_error "Insufficient disk space: ${disk_usage}% used"
        return 1
    fi
    
    log_deploy_success "Prerequisites validated successfully"
    return 0
}

# Pre-deployment health check
pre_deployment_health_check() {
    local environment="$1"
    
    log_deploy "Running pre-deployment health check for $environment..."
    
    # Check if current application is healthy
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_deploy_success "Current application is healthy"
    else
        log_deploy_warning "Current application health check failed, proceeding with deployment"
    fi
    
    # Check system resources
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' 2>/dev/null || echo "0")
    local memory_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}' 2>/dev/null || echo "0")
    
    if [[ $cpu_usage -gt 90 ]] || [[ $memory_usage -gt 90 ]]; then
        log_deploy_warning "High resource usage detected: CPU=${cpu_usage}%, Memory=${memory_usage}%"
    fi
    
    log_deploy_success "Pre-deployment health check completed"
}

# Blue-Green deployment
deploy_blue_green() {
    local environment="$1"
    local version="$2"
    local image_tag="$3"
    
    log_deploy "Starting Blue-Green deployment to $environment..."
    
    # Determine current color
    local current_color="blue"
    if docker ps --format "table {{.Names}}" | grep -q "opinion-market-green"; then
        current_color="green"
    fi
    
    local new_color="green"
    if [[ "$current_color" == "green" ]]; then
        new_color="blue"
    fi
    
    log_deploy "Current color: $current_color, Deploying to: $new_color"
    
    # Deploy to new color
    local new_container_name="opinion-market-$new_color"
    local new_port=$((8000 + (new_color == "green" ? 1 : 0)))
    
    # Stop existing container of new color
    docker stop "$new_container_name" 2>/dev/null || true
    docker rm "$new_container_name" 2>/dev/null || true
    
    # Start new container
    docker run -d \
        --name "$new_container_name" \
        -p "$new_port:8000" \
        --restart unless-stopped \
        "opinion-market:$image_tag"
    
    # Wait for new container to be ready
    local retry_count=0
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        if curl -f "http://localhost:$new_port/health" &> /dev/null; then
            log_deploy_success "New $new_color container is healthy"
            break
        fi
        
        retry_count=$((retry_count + 1))
        log_deploy "Waiting for new container to be ready... (attempt $retry_count/$MAX_RETRIES)"
        sleep 10
    done
    
    if [[ $retry_count -eq $MAX_RETRIES ]]; then
        log_deploy_error "New container failed to become healthy"
        return 1
    fi
    
    # Switch traffic (update load balancer or proxy configuration)
    switch_traffic "$new_color" "$new_port"
    
    # Stop old container after traffic switch
    local old_container_name="opinion-market-$current_color"
    docker stop "$old_container_name" 2>/dev/null || true
    
    log_deploy_success "Blue-Green deployment completed successfully"
    return 0
}

# Rolling deployment
deploy_rolling() {
    local environment="$1"
    local version="$2"
    local image_tag="$3"
    
    log_deploy "Starting Rolling deployment to $environment..."
    
    # For Docker Compose
    if [[ -f "docker-compose.$environment.yml" ]]; then
        # Update image in docker-compose file
        sed -i "s|image: opinion-market:.*|image: opinion-market:$image_tag|" "docker-compose.$environment.yml"
        
        # Rolling update
        docker-compose -f "docker-compose.$environment.yml" up -d --no-deps --scale app=2
        
        # Wait for new instances to be healthy
        sleep 30
        
        # Scale down old instances
        docker-compose -f "docker-compose.$environment.yml" up -d --no-deps --scale app=1
        
        log_deploy_success "Rolling deployment completed successfully"
        return 0
    fi
    
    # For Kubernetes
    if command -v kubectl &> /dev/null; then
        kubectl set image deployment/opinion-market app="opinion-market:$image_tag"
        kubectl rollout status deployment/opinion-market --timeout=300s
        
        log_deploy_success "Kubernetes rolling deployment completed successfully"
        return 0
    fi
    
    # Fallback to simple restart
    docker stop opinion-market 2>/dev/null || true
    docker rm opinion-market 2>/dev/null || true
    docker run -d \
        --name opinion-market \
        -p 8000:8000 \
        --restart unless-stopped \
        "opinion-market:$image_tag"
    
    log_deploy_success "Simple rolling deployment completed successfully"
    return 0
}

# Canary deployment
deploy_canary() {
    local environment="$1"
    local version="$2"
    local image_tag="$3"
    local canary_percentage="${4:-10}"
    
    log_deploy "Starting Canary deployment to $environment (${canary_percentage}% traffic)..."
    
    # Deploy canary version
    local canary_container_name="opinion-market-canary"
    local canary_port=8001
    
    # Stop existing canary
    docker stop "$canary_container_name" 2>/dev/null || true
    docker rm "$canary_container_name" 2>/dev/null || true
    
    # Start canary container
    docker run -d \
        --name "$canary_container_name" \
        -p "$canary_port:8000" \
        --restart unless-stopped \
        "opinion-market:$image_tag"
    
    # Wait for canary to be ready
    local retry_count=0
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        if curl -f "http://localhost:$canary_port/health" &> /dev/null; then
            log_deploy_success "Canary container is healthy"
            break
        fi
        
        retry_count=$((retry_count + 1))
        log_deploy "Waiting for canary container to be ready... (attempt $retry_count/$MAX_RETRIES)"
        sleep 10
    done
    
    if [[ $retry_count -eq $MAX_RETRIES ]]; then
        log_deploy_error "Canary container failed to become healthy"
        return 1
    fi
    
    # Route percentage of traffic to canary
    route_canary_traffic "$canary_percentage" "$canary_port"
    
    log_deploy_success "Canary deployment completed successfully"
    return 0
}

# Switch traffic for blue-green deployment
switch_traffic() {
    local new_color="$1"
    local new_port="$2"
    
    log_deploy "Switching traffic to $new_color container on port $new_port..."
    
    # Update nginx configuration (if using nginx)
    if [[ -f "/etc/nginx/sites-available/opinion-market" ]]; then
        sed -i "s|proxy_pass http://localhost:[0-9]*|proxy_pass http://localhost:$new_port|" "/etc/nginx/sites-available/opinion-market"
        nginx -s reload
        log_deploy_success "Nginx configuration updated"
    fi
    
    # Update load balancer configuration
    # This would depend on your load balancer setup
    
    log_deploy_success "Traffic switched to $new_color"
}

# Route canary traffic
route_canary_traffic() {
    local percentage="$1"
    local canary_port="$2"
    
    log_deploy "Routing ${percentage}% of traffic to canary on port $canary_port..."
    
    # Update nginx configuration for canary routing
    if [[ -f "/etc/nginx/sites-available/opinion-market" ]]; then
        # Add upstream configuration for canary
        cat >> "/etc/nginx/sites-available/opinion-market" << EOF
upstream opinion-market {
    server localhost:8000 weight=$((100 - percentage));
    server localhost:$canary_port weight=$percentage;
}
EOF
        nginx -s reload
        log_deploy_success "Canary traffic routing configured"
    fi
}

# Post-deployment health check
post_deployment_health_check() {
    local environment="$1"
    local version="$2"
    local timeout="$3"
    
    log_deploy "Running post-deployment health check for $environment..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        # Check application health
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_deploy_success "Application health check passed"
            
            # Check API endpoints
            if curl -f http://localhost:8000/api/v1/health &> /dev/null; then
                log_deploy_success "API health check passed"
                
                # Check metrics endpoint
                if curl -f http://localhost:8000/metrics &> /dev/null; then
                    log_deploy_success "Metrics endpoint check passed"
                    return 0
                else
                    log_deploy_warning "Metrics endpoint check failed"
                fi
            else
                log_deploy_warning "API health check failed"
            fi
        else
            log_deploy_warning "Application health check failed"
        fi
        
        sleep 10
    done
    
    log_deploy_error "Post-deployment health check timed out"
    return 1
}

# Rollback deployment
rollback_deployment() {
    local environment="$1"
    local rollback_version="$2"
    
    log_deploy "Starting rollback to version $rollback_version in $environment..."
    
    # Get rollback version from deployment history
    if [[ -z "$rollback_version" ]]; then
        rollback_version=$(jq -r '.rollback_version' "$DEPLOYMENT_HISTORY")
    fi
    
    if [[ "$rollback_version" == "null" ]] || [[ -z "$rollback_version" ]]; then
        log_deploy_error "No rollback version available"
        return 1
    fi
    
    log_deploy "Rolling back to version: $rollback_version"
    
    # Stop current deployment
    docker stop opinion-market 2>/dev/null || true
    docker rm opinion-market 2>/dev/null || true
    
    # Deploy rollback version
    docker run -d \
        --name opinion-market \
        -p 8000:8000 \
        --restart unless-stopped \
        "opinion-market:$rollback_version"
    
    # Wait for rollback to be healthy
    local retry_count=0
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_deploy_success "Rollback deployment is healthy"
            break
        fi
        
        retry_count=$((retry_count + 1))
        log_deploy "Waiting for rollback to be ready... (attempt $retry_count/$MAX_RETRIES)"
        sleep 10
    done
    
    if [[ $retry_count -eq $MAX_RETRIES ]]; then
        log_deploy_error "Rollback deployment failed to become healthy"
        return 1
    fi
    
    # Update deployment history
    update_deployment_history "$environment" "$rollback_version" "rollback"
    
    log_deploy_success "Rollback completed successfully"
    return 0
}

# Update deployment history
update_deployment_history() {
    local environment="$1"
    local version="$2"
    local deployment_type="$3"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Create new deployment record
    local new_deployment=$(cat << EOF
{
    "timestamp": "$timestamp",
    "environment": "$environment",
    "version": "$version",
    "type": "$deployment_type",
    "status": "success"
}
EOF
)
    
    # Update deployment history
    jq --argjson new_deployment "$new_deployment" '
        .deployments += [$new_deployment] |
        .current_version = $new_deployment.version |
        if .deployments | length > 1 then
            .rollback_version = (.deployments[-2].version)
        else
            .
        end
    ' "$DEPLOYMENT_HISTORY" > "${DEPLOYMENT_HISTORY}.tmp" && mv "${DEPLOYMENT_HISTORY}.tmp" "$DEPLOYMENT_HISTORY"
    
    log_deploy "Deployment history updated"
}

# Cleanup deployment resources
cleanup_deployment() {
    log_deploy "Cleaning up deployment resources..."
    
    # Remove temporary files
    rm -f "${DEPLOYMENT_HISTORY}.tmp"
    
    # Clean up old containers (keep last 3 versions)
    docker images --format "table {{.Repository}}:{{.Tag}}" | grep "opinion-market" | tail -n +4 | xargs -r docker rmi
    
    log_deploy "Deployment cleanup completed"
}

# Main deployment function
deploy() {
    local environment="$1"
    local version="$2"
    local strategy="${3:-$ROLLING}"
    local image_tag="${4:-$version}"
    
    log_deploy "Starting deployment to $environment with strategy $strategy..."
    
    # Initialize deployment system
    init_deployment
    
    # Validate prerequisites
    if ! validate_prerequisites "$environment" "$version"; then
        log_deploy_error "Prerequisites validation failed"
        return 1
    fi
    
    # Pre-deployment health check
    pre_deployment_health_check "$environment"
    
    # Deploy based on strategy
    case "$strategy" in
        "$BLUE_GREEN")
            if ! deploy_blue_green "$environment" "$version" "$image_tag"; then
                log_deploy_error "Blue-Green deployment failed"
                return 1
            fi
            ;;
        "$ROLLING")
            if ! deploy_rolling "$environment" "$version" "$image_tag"; then
                log_deploy_error "Rolling deployment failed"
                return 1
            fi
            ;;
        "$CANARY")
            if ! deploy_canary "$environment" "$version" "$image_tag"; then
                log_deploy_error "Canary deployment failed"
                return 1
            fi
            ;;
        *)
            log_deploy_error "Unknown deployment strategy: $strategy"
            return 1
            ;;
    esac
    
    # Post-deployment health check
    if ! post_deployment_health_check "$environment" "$version" "$HEALTH_CHECK_TIMEOUT"; then
        log_deploy_error "Post-deployment health check failed, initiating rollback..."
        rollback_deployment "$environment"
        return 1
    fi
    
    # Update deployment history
    update_deployment_history "$environment" "$version" "$strategy"
    
    log_deploy_success "Deployment to $environment completed successfully!"
    return 0
}

# Show deployment status
show_status() {
    echo -e "${PURPLE}🚀 Deployment Status${NC}"
    echo ""
    
    # Current deployment
    if [[ -f "$DEPLOYMENT_HISTORY" ]]; then
        local current_version=$(jq -r '.current_version' "$DEPLOYMENT_HISTORY")
        local rollback_version=$(jq -r '.rollback_version' "$DEPLOYMENT_HISTORY")
        
        echo -e "${GREEN}Current Version:${NC} $current_version"
        echo -e "${YELLOW}Rollback Version:${NC} $rollback_version"
        echo ""
        
        # Recent deployments
        echo -e "${BLUE}Recent Deployments:${NC}"
        jq -r '.deployments[-5:] | .[] | "\(.timestamp) - \(.environment) - \(.version) - \(.type)"' "$DEPLOYMENT_HISTORY"
    fi
    
    echo ""
    
    # Container status
    echo -e "${BLUE}Container Status:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep opinion-market || echo "No opinion-market containers running"
    
    echo ""
    
    # Health status
    echo -e "${BLUE}Health Status:${NC}"
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo -e "Application: ${GREEN}HEALTHY${NC}"
    else
        echo -e "Application: ${RED}UNHEALTHY${NC}"
    fi
}

# Help function
show_help() {
    echo "Automated Deployment & Rollback System"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy ENVIRONMENT VERSION [STRATEGY] [IMAGE_TAG]  Deploy application"
    echo "  rollback ENVIRONMENT [VERSION]                     Rollback deployment"
    echo "  status                                             Show deployment status"
    echo "  health                                             Check application health"
    echo "  help                                               Show this help message"
    echo ""
    echo "Environments:"
    echo "  staging, production"
    echo ""
    echo "Strategies:"
    echo "  rolling     (default) - Rolling deployment"
    echo "  blue-green           - Blue-Green deployment"
    echo "  canary               - Canary deployment"
    echo ""
    echo "Examples:"
    echo "  $0 deploy staging v1.2.3"
    echo "  $0 deploy production v1.2.3 blue-green"
    echo "  $0 rollback production"
    echo "  $0 status"
}

# Check application health
check_health() {
    echo -e "${PURPLE}🏥 Application Health Check${NC}"
    echo ""
    
    # Check main health endpoint
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo -e "Main Health: ${GREEN}✅ HEALTHY${NC}"
    else
        echo -e "Main Health: ${RED}❌ UNHEALTHY${NC}"
    fi
    
    # Check API health endpoint
    if curl -f http://localhost:8000/api/v1/health &> /dev/null; then
        echo -e "API Health: ${GREEN}✅ HEALTHY${NC}"
    else
        echo -e "API Health: ${RED}❌ UNHEALTHY${NC}"
    fi
    
    # Check metrics endpoint
    if curl -f http://localhost:8000/metrics &> /dev/null; then
        echo -e "Metrics: ${GREEN}✅ HEALTHY${NC}"
    else
        echo -e "Metrics: ${RED}❌ UNHEALTHY${NC}"
    fi
    
    # Check response time
    local response_time=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8000/ 2>/dev/null || echo "N/A")
    echo -e "Response Time: ${response_time}s"
}

# Main function
main() {
    case "${1:-}" in
        deploy)
            if [[ $# -lt 3 ]]; then
                echo "Error: deploy command requires ENVIRONMENT and VERSION"
                show_help
                exit 1
            fi
            deploy "$2" "$3" "${4:-$ROLLING}" "${5:-$3}"
            ;;
        rollback)
            if [[ $# -lt 2 ]]; then
                echo "Error: rollback command requires ENVIRONMENT"
                show_help
                exit 1
            fi
            rollback_deployment "$2" "${3:-}"
            ;;
        status)
            show_status
            ;;
        health)
            check_health
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            show_help
            ;;
        *)
            echo "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
