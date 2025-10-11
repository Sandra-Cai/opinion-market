#!/bin/bash

# Opinion Market Deployment Script
# This script handles the deployment of the Opinion Market application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
COMPOSE_FILE="docker-compose.yml"
BACKUP_DIR="./backups"
LOG_DIR="./logs"

# Functions
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

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if required files exist
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_success "All dependencies are satisfied"
}

create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "./uploads"
    mkdir -p "./models"
    mkdir -p "./nginx/ssl"
    
    log_success "Directories created"
}

generate_ssl_certificates() {
    log_info "Generating SSL certificates..."
    
    if [ ! -f "./nginx/ssl/cert.pem" ] || [ ! -f "./nginx/ssl/key.pem" ]; then
        log_warning "SSL certificates not found. Generating self-signed certificates..."
        
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ./nginx/ssl/key.pem \
            -out ./nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        log_success "Self-signed SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

backup_database() {
    if [ "$ENVIRONMENT" = "production" ]; then
        log_info "Creating database backup..."
        
        BACKUP_FILE="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql"
        
        docker-compose exec -T postgres pg_dump -U postgres opinion_market > "$BACKUP_FILE"
        
        if [ $? -eq 0 ]; then
            log_success "Database backup created: $BACKUP_FILE"
        else
            log_error "Failed to create database backup"
            exit 1
        fi
    fi
}

build_images() {
    log_info "Building Docker images..."
    
    docker-compose build --no-cache
    
    if [ $? -eq 0 ]; then
        log_success "Docker images built successfully"
    else
        log_error "Failed to build Docker images"
        exit 1
    fi
}

run_migrations() {
    log_info "Running database migrations..."
    
    docker-compose run --rm migration
    
    if [ $? -eq 0 ]; then
        log_success "Database migrations completed"
    else
        log_error "Database migrations failed"
        exit 1
    fi
}

start_services() {
    log_info "Starting services..."
    
    if [ "$ENVIRONMENT" = "development" ]; then
        docker-compose --profile dev up -d
    else
        docker-compose up -d
    fi
    
    if [ $? -eq 0 ]; then
        log_success "Services started successfully"
    else
        log_error "Failed to start services"
        exit 1
    fi
}

wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for database
    log_info "Waiting for database..."
    until docker-compose exec postgres pg_isready -U postgres -d opinion_market; do
        sleep 2
    done
    log_success "Database is ready"
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    until docker-compose exec redis redis-cli ping; do
        sleep 2
    done
    log_success "Redis is ready"
    
    # Wait for API
    log_info "Waiting for API..."
    until curl -f http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5
    done
    log_success "API is ready"
}

check_health() {
    log_info "Checking service health..."
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        exit 1
    fi
    
    # Check database health
    if docker-compose exec postgres pg_isready -U postgres -d opinion_market > /dev/null 2>&1; then
        log_success "Database health check passed"
    else
        log_error "Database health check failed"
        exit 1
    fi
    
    # Check Redis health
    if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis health check passed"
    else
        log_error "Redis health check failed"
        exit 1
    fi
}

show_status() {
    log_info "Deployment Status:"
    echo ""
    echo "Services:"
    docker-compose ps
    echo ""
    echo "API Health:"
    curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
    echo ""
    echo "Access URLs:"
    echo "  API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  Grafana: http://localhost:3000 (admin/admin_password)"
    echo "  Prometheus: http://localhost:9090"
    echo ""
}

cleanup() {
    log_info "Cleaning up old containers and images..."
    
    # Remove stopped containers
    docker-compose rm -f
    
    # Remove unused images
    docker image prune -f
    
    log_success "Cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment for environment: $ENVIRONMENT"
    
    check_dependencies
    create_directories
    generate_ssl_certificates
    
    if [ "$ENVIRONMENT" = "production" ]; then
        backup_database
    fi
    
    build_images
    run_migrations
    start_services
    wait_for_services
    check_health
    show_status
    
    log_success "Deployment completed successfully!"
}

# Rollback function
rollback() {
    log_info "Starting rollback..."
    
    # Stop current services
    docker-compose down
    
    # Restore from latest backup
    LATEST_BACKUP=$(ls -t $BACKUP_DIR/*.sql 2>/dev/null | head -n1)
    
    if [ -n "$LATEST_BACKUP" ]; then
        log_info "Restoring from backup: $LATEST_BACKUP"
        
        # Start database
        docker-compose up -d postgres
        sleep 10
        
        # Restore database
        docker-compose exec -T postgres psql -U postgres -d opinion_market < "$LATEST_BACKUP"
        
        if [ $? -eq 0 ]; then
            log_success "Database restored successfully"
        else
            log_error "Failed to restore database"
            exit 1
        fi
    else
        log_warning "No backup found for rollback"
    fi
    
    # Start services
    start_services
    wait_for_services
    
    log_success "Rollback completed"
}

# Update function
update() {
    log_info "Starting update..."
    
    # Pull latest images
    docker-compose pull
    
    # Restart services
    docker-compose up -d --force-recreate
    
    # Run migrations
    run_migrations
    
    wait_for_services
    check_health
    
    log_success "Update completed"
}

# Main script logic
case "${2:-deploy}" in
    "deploy")
        deploy
        ;;
    "rollback")
        rollback
        ;;
    "update")
        update
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        show_status
        ;;
    *)
        echo "Usage: $0 [environment] [action]"
        echo ""
        echo "Environments:"
        echo "  production  - Production deployment (default)"
        echo "  development - Development deployment"
        echo ""
        echo "Actions:"
        echo "  deploy   - Deploy the application (default)"
        echo "  rollback - Rollback to previous version"
        echo "  update   - Update to latest version"
        echo "  cleanup  - Clean up old containers and images"
        echo "  status   - Show deployment status"
        echo ""
        echo "Examples:"
        echo "  $0 production deploy"
        echo "  $0 development deploy"
        echo "  $0 production rollback"
        echo "  $0 production update"
        exit 1
        ;;
esac