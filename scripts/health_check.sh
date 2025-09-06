#!/bin/bash

# 🏥 Comprehensive CI/CD Health Check Script
# Monitors the health of the Opinion Market CI/CD pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${PURPLE}"
    echo "=========================================="
    echo "🏥 CI/CD Health Check Dashboard"
    echo "=========================================="
    echo -e "${NC}"
}

# Health check counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Health check function
run_health_check() {
    local check_name="$1"
    local check_command="$2"
    local check_description="$3"
    local severity="${4:-error}"  # error, warning, info
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    log_info "Checking: $check_name"
    echo "Description: $check_description"
    
    if eval "$check_command"; then
        log_success "$check_name passed"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        case $severity in
            "error")
                log_error "$check_name failed"
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
                ;;
            "warning")
                log_warning "$check_name failed (warning)"
                WARNING_CHECKS=$((WARNING_CHECKS + 1))
                ;;
            "info")
                log_info "$check_name failed (info)"
                ;;
        esac
    fi
    echo ""
}

# Main health check function
main() {
    log_header
    
    # 1. Environment Health Checks
    log_info "🔧 Environment Health Checks"
    echo "=========================================="
    
    run_health_check "Python Version" \
        "python --version | grep -E 'Python 3\.(11|12)'" \
        "Verify Python version is 3.11 or 3.12" \
        "error"
    
    run_health_check "Pip Installation" \
        "pip --version" \
        "Verify pip is available and working" \
        "error"
    
    run_health_check "Docker Installation" \
        "docker --version && docker info >/dev/null 2>&1" \
        "Verify Docker is installed and running" \
        "warning"
    
    run_health_check "Git Configuration" \
        "git config --get user.name && git config --get user.email" \
        "Verify Git is configured" \
        "warning"
    
    # 2. Project Structure Health Checks
    log_info "📁 Project Structure Health Checks"
    echo "=========================================="
    
    run_health_check "Critical Directories" \
        "ls -d app/ tests/ .github/workflows/ deployment/ 2>/dev/null" \
        "Verify critical project directories exist" \
        "error"
    
    run_health_check "Critical Files" \
        "ls requirements.txt app/main_simple.py Dockerfile.robust 2>/dev/null" \
        "Verify critical project files exist" \
        "error"
    
    run_health_check "GitHub Actions Workflows" \
        "ls .github/workflows/*.yml 2>/dev/null | wc -l | grep -E '[5-9]|[0-9]{2,}'" \
        "Verify sufficient number of GitHub Actions workflows" \
        "warning"
    
    # 3. Dependencies Health Checks
    log_info "📦 Dependencies Health Checks"
    echo "=========================================="
    
    run_health_check "Core Python Dependencies" \
        "python -c 'import fastapi, uvicorn, pytest, httpx; print(\"All core deps available\")'" \
        "Verify core Python dependencies are importable" \
        "error"
    
    run_health_check "Requirements File Validity" \
        "pip install --dry-run -r requirements.txt >/dev/null 2>&1" \
        "Verify requirements.txt is valid" \
        "error"
    
    run_health_check "Development Dependencies" \
        "pip install --dry-run -r requirements-dev.txt >/dev/null 2>&1" \
        "Verify requirements-dev.txt is valid" \
        "warning"
    
    # 4. Application Health Checks
    log_info "🚀 Application Health Checks"
    echo "=========================================="
    
    run_health_check "Simple App Import" \
        "python -c 'from app.main_simple import app; print(\"Simple app imported\")'" \
        "Verify simple app can be imported" \
        "error"
    
    run_health_check "Main App Import" \
        "python -c 'from app.main import app; print(\"Main app imported\")'" \
        "Verify main app can be imported" \
        "warning"
    
    run_health_check "FastAPI App Creation" \
        "python -c 'from fastapi import FastAPI; app = FastAPI(); print(\"FastAPI app created\")'" \
        "Verify FastAPI can create applications" \
        "error"
    
    # 5. Testing Health Checks
    log_info "🧪 Testing Health Checks"
    echo "=========================================="
    
    run_health_check "Pytest Installation" \
        "pytest --version" \
        "Verify pytest is installed and working" \
        "error"
    
    run_health_check "Test Discovery" \
        "pytest --collect-only -q 2>/dev/null | grep -E 'test session starts|collected [0-9]+ items'" \
        "Verify pytest can discover tests" \
        "error"
    
    run_health_check "Basic Test Execution" \
        "pytest tests/test_simple_app.py -v --tb=short >/dev/null 2>&1" \
        "Verify basic tests can run" \
        "error"
    
    run_health_check "Robust Test Execution" \
        "pytest tests/test_robust.py -v --tb=short >/dev/null 2>&1" \
        "Verify robust tests can run" \
        "warning"
    
    # 6. Code Quality Health Checks
    log_info "📝 Code Quality Health Checks"
    echo "=========================================="
    
    run_health_check "Black Installation" \
        "black --version" \
        "Verify Black formatter is installed" \
        "warning"
    
    run_health_check "Flake8 Installation" \
        "flake8 --version" \
        "Verify Flake8 linter is installed" \
        "warning"
    
    run_health_check "Isort Installation" \
        "isort --version" \
        "Verify Isort import sorter is installed" \
        "warning"
    
    run_health_check "MyPy Installation" \
        "mypy --version" \
        "Verify MyPy type checker is installed" \
        "warning"
    
    # 7. Security Health Checks
    log_info "🔒 Security Health Checks"
    echo "=========================================="
    
    run_health_check "Bandit Installation" \
        "bandit --version" \
        "Verify Bandit security linter is installed" \
        "warning"
    
    run_health_check "Safety Installation" \
        "safety --version" \
        "Verify Safety dependency checker is installed" \
        "warning"
    
    run_health_check "Basic Security Scan" \
        "bandit -r app/ -f json -o /tmp/bandit-report.json >/dev/null 2>&1" \
        "Verify Bandit can scan the codebase" \
        "warning"
    
    # 8. Docker Health Checks
    log_info "🐳 Docker Health Checks"
    echo "=========================================="
    
    run_health_check "Docker Build Test (Simple)" \
        "docker build -f Dockerfile.simple -t health-check-simple . >/dev/null 2>&1" \
        "Verify simple Dockerfile can build" \
        "warning"
    
    run_health_check "Docker Build Test (Robust)" \
        "docker build -f Dockerfile.robust -t health-check-robust . --target base >/dev/null 2>&1" \
        "Verify robust Dockerfile can build" \
        "warning"
    
    run_health_check "Docker Container Test" \
        "docker run --rm -d --name health-check-container -p 8002:8000 health-check-simple >/dev/null 2>&1 && sleep 5 && curl -f http://localhost:8002/health >/dev/null 2>&1 && docker stop health-check-container >/dev/null 2>&1" \
        "Verify Docker container can run and respond" \
        "warning"
    
    # 9. CI/CD Configuration Health Checks
    log_info "🔄 CI/CD Configuration Health Checks"
    echo "=========================================="
    
    run_health_check "YAML Syntax Validation" \
        "find .github/workflows/ -name '*.yml' -exec python -c 'import yaml; yaml.safe_load(open(\"{}\"))' \\; >/dev/null 2>&1" \
        "Verify all GitHub Actions YAML files are valid" \
        "error"
    
    run_health_check "Workflow Triggers" \
        "grep -r 'on:' .github/workflows/ | grep -E '(push|pull_request|workflow_dispatch)'" \
        "Verify workflows have proper triggers" \
        "warning"
    
    run_health_check "Environment Variables" \
        "grep -r 'env:' .github/workflows/ | grep -E '(PYTHON_VERSION|REGISTRY|IMAGE_NAME)'" \
        "Verify workflows define necessary environment variables" \
        "warning"
    
    # 10. Performance Health Checks
    log_info "⚡ Performance Health Checks"
    echo "=========================================="
    
    run_health_check "Import Performance" \
        "time python -c 'from app.main_simple import app' 2>&1 | grep -E 'real.*[0-9]m[0-9]\\.[0-9]s' | awk '{print \$2}' | sed 's/m/ /' | awk '{if (\$1 < 1) exit 0; else exit 1}'" \
        "Verify app imports in under 1 second" \
        "warning"
    
    run_health_check "Test Execution Performance" \
        "time pytest tests/test_simple_app.py -v --tb=short 2>&1 | grep -E 'real.*[0-9]m[0-9]\\.[0-9]s' | awk '{print \$2}' | sed 's/m/ /' | awk '{if (\$1 < 2) exit 0; else exit 1}'" \
        "Verify basic tests complete in under 2 seconds" \
        "warning"
    
    # 11. Documentation Health Checks
    log_info "📚 Documentation Health Checks"
    echo "=========================================="
    
    run_health_check "README Files" \
        "ls README*.md ROBUST_CI_CD_README.md 2>/dev/null" \
        "Verify documentation files exist" \
        "warning"
    
    run_health_check "Code Documentation" \
        "find app/ -name '*.py' -exec grep -l 'def\\|class' {} \\; | head -5 | xargs grep -l 'def\\|class'" \
        "Verify code files have functions/classes" \
        "info"
    
    # 12. Network Health Checks
    log_info "🌐 Network Health Checks"
    echo "=========================================="
    
    run_health_check "GitHub Connectivity" \
        "curl -f https://api.github.com >/dev/null 2>&1" \
        "Verify GitHub API is accessible" \
        "warning"
    
    run_health_check "PyPI Connectivity" \
        "curl -f https://pypi.org/simple/ >/dev/null 2>&1" \
        "Verify PyPI is accessible" \
        "warning"
    
    run_health_check "Docker Hub Connectivity" \
        "curl -f https://hub.docker.com >/dev/null 2>&1" \
        "Verify Docker Hub is accessible" \
        "warning"
    
    # Summary
    log_info "📊 Health Check Summary"
    echo "=========================================="
    echo "Total Checks: $TOTAL_CHECKS"
    echo "Passed: $PASSED_CHECKS"
    echo "Failed: $FAILED_CHECKS"
    echo "Warnings: $WARNING_CHECKS"
    echo ""
    
    # Overall health status
    if [ $FAILED_CHECKS -eq 0 ]; then
        if [ $WARNING_CHECKS -eq 0 ]; then
            log_success "🎉 All health checks passed! CI/CD pipeline is in excellent health!"
            echo -e "${GREEN}🚀 Ready for production deployment!${NC}"
            exit 0
        else
            log_warning "⚠️  Health checks passed with warnings. CI/CD pipeline is functional but could be improved."
            echo -e "${YELLOW}🔧 Consider addressing the warnings for optimal performance.${NC}"
            exit 0
        fi
    else
        log_error "❌ Some critical health checks failed. CI/CD pipeline needs attention."
        echo -e "${RED}🛠️  Please fix the failed checks before proceeding.${NC}"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log_info "🧹 Cleaning up health check artifacts..."
    
    # Stop any running containers
    docker stop health-check-container 2>/dev/null || true
    docker rm health-check-container 2>/dev/null || true
    
    # Remove test images
    docker rmi health-check-simple health-check-robust 2>/dev/null || true
    
    # Remove temporary files
    rm -f /tmp/bandit-report.json 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Set up trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"
