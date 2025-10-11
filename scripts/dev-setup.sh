#!/bin/bash

# Opinion Market Development Setup Script
# This script sets up the development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

check_python() {
    log_info "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.11 or later."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.11"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        log_error "Python 3.11 or later is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    
    log_success "Python $PYTHON_VERSION is installed"
}

check_pip() {
    log_info "Checking pip installation..."
    
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed. Please install pip3."
        exit 1
    fi
    
    log_success "pip3 is installed"
}

create_virtual_environment() {
    log_info "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
}

activate_virtual_environment() {
    log_info "Activating virtual environment..."
    
    source venv/bin/activate
    
    if [ $? -eq 0 ]; then
        log_success "Virtual environment activated"
    else
        log_error "Failed to activate virtual environment"
        exit 1
    fi
}

install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    pip install -r requirements-dev.txt
    
    if [ $? -eq 0 ]; then
        log_success "Dependencies installed successfully"
    else
        log_error "Failed to install dependencies"
        exit 1
    fi
}

setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    if [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        
        if [ $? -eq 0 ]; then
            log_success "Pre-commit hooks installed"
        else
            log_warning "Failed to install pre-commit hooks"
        fi
    else
        log_info "No pre-commit configuration found"
    fi
}

create_env_file() {
    log_info "Creating environment file..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=sqlite:///./dev.db

# Redis (optional for development)
REDIS_URL=redis://localhost:6379/0
ENABLE_CACHING=false

# Security
SECRET_KEY=dev-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS
ALLOWED_HOSTS=["http://localhost:3000", "http://localhost:8000"]

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# Caching
CACHE_TTL=300

# Performance
ENABLE_COMPRESSION=true
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# WebSocket
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=100

# ML
ML_ENABLED=false
ML_MODEL_PATH=./models

# Blockchain
BLOCKCHAIN_ENABLED=false

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# Email (optional)
SMTP_HOST=localhost
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_TLS=true

# File Storage
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760

# Backup
BACKUP_ENABLED=false
BACKUP_RETENTION_DAYS=30
EOF
        
        log_success "Environment file created"
    else
        log_info "Environment file already exists"
    fi
}

create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p uploads
    mkdir -p models
    mkdir -p backups
    
    log_success "Directories created"
}

run_tests() {
    log_info "Running tests..."
    
    pytest tests/ -v --cov=app --cov-report=html --cov-report=term
    
    if [ $? -eq 0 ]; then
        log_success "All tests passed"
    else
        log_warning "Some tests failed"
    fi
}

run_linting() {
    log_info "Running code quality checks..."
    
    # Black formatting
    log_info "Running Black formatter..."
    black app/ tests/ --check
    
    # isort import sorting
    log_info "Running isort..."
    isort app/ tests/ --check-only
    
    # flake8 linting
    log_info "Running flake8..."
    flake8 app/ tests/
    
    # mypy type checking
    log_info "Running mypy..."
    mypy app/
    
    # bandit security check
    log_info "Running bandit security check..."
    bandit -r app/
    
    log_success "Code quality checks completed"
}

show_development_info() {
    log_info "Development Environment Setup Complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Start the development server:"
    echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    echo "3. Or use Docker for development:"
    echo "   docker-compose --profile dev up -d"
    echo ""
    echo "4. Access the application:"
    echo "   API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo "   ReDoc: http://localhost:8000/redoc"
    echo ""
    echo "5. Run tests:"
    echo "   pytest tests/ -v"
    echo ""
    echo "6. Run code quality checks:"
    echo "   ./scripts/dev-setup.sh lint"
    echo ""
    echo "7. Format code:"
    echo "   black app/ tests/"
    echo "   isort app/ tests/"
    echo ""
}

# Main setup function
setup() {
    log_info "Setting up Opinion Market development environment..."
    
    check_python
    check_pip
    create_virtual_environment
    activate_virtual_environment
    install_dependencies
    setup_pre_commit
    create_env_file
    create_directories
    
    log_success "Development environment setup completed!"
    show_development_info
}

# Main script logic
case "${1:-setup}" in
    "setup")
        setup
        ;;
    "test")
        activate_virtual_environment
        run_tests
        ;;
    "lint")
        activate_virtual_environment
        run_linting
        ;;
    "format")
        activate_virtual_environment
        log_info "Formatting code..."
        black app/ tests/
        isort app/ tests/
        log_success "Code formatted"
        ;;
    "clean")
        log_info "Cleaning up development environment..."
        rm -rf venv
        rm -rf __pycache__
        rm -rf .pytest_cache
        rm -rf .coverage
        rm -rf htmlcov
        rm -rf .mypy_cache
        rm -rf .bandit
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        log_success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  setup   - Set up development environment (default)"
        echo "  test    - Run tests"
        echo "  lint    - Run code quality checks"
        echo "  format  - Format code with Black and isort"
        echo "  clean   - Clean up development environment"
        echo ""
        echo "Examples:"
        echo "  $0 setup"
        echo "  $0 test"
        echo "  $0 lint"
        echo "  $0 format"
        echo "  $0 clean"
        exit 1
        ;;
esac


