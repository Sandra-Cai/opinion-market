#!/bin/bash

# 🚀 CI/CD Pipeline Monitoring Script
# Comprehensive monitoring and alerting for the Opinion Market CI/CD pipeline

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
LOG_FILE="/tmp/cicd_monitor.log"
ALERT_THRESHOLD=5
MAX_RETRIES=3
HEALTH_CHECK_INTERVAL=30

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

# Error logging
log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Success logging
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Warning logging
log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Header
print_header() {
    echo -e "${PURPLE}"
    echo "=========================================="
    echo "🚀 CI/CD Pipeline Monitor"
    echo "=========================================="
    echo -e "${NC}"
}

# Check GitHub Actions status
check_github_actions() {
    log "Checking GitHub Actions status..."
    
    if command -v gh &> /dev/null; then
        # Get latest workflow runs
        local runs=$(gh run list --limit 5 --json status,conclusion,createdAt,workflowName)
        
        if [ $? -eq 0 ]; then
            log_success "GitHub Actions status retrieved"
            echo "$runs" | jq -r '.[] | "\(.workflowName): \(.status) - \(.conclusion // "running")"'
        else
            log_warning "Could not retrieve GitHub Actions status (gh CLI not authenticated)"
        fi
    else
        log_warning "GitHub CLI not installed, skipping GitHub Actions check"
    fi
}

# Check application health
check_app_health() {
    log "Checking application health..."
    
    # Test simple app import
    if python -c "from app.main_simple import app; print('App imported successfully')" 2>/dev/null; then
        log_success "Simple app import: OK"
    else
        log_error "Simple app import: FAILED"
        return 1
    fi
    
    # Test models import
    if python -c "import app.models; print('Models imported successfully')" 2>/dev/null; then
        log_success "Models import: OK"
    else
        log_error "Models import: FAILED"
        return 1
    fi
    
    # Test services import
    if python -c "import app.services.mobile_api; print('Services imported successfully')" 2>/dev/null; then
        log_success "Services import: OK"
    else
        log_error "Services import: FAILED"
        return 1
    fi
}

# Check test status
check_tests() {
    log "Running quick test suite..."
    
    # Run simple app tests
    if python -m pytest tests/test_simple_app.py -v --tb=short 2>/dev/null; then
        log_success "Simple app tests: PASSED"
    else
        log_error "Simple app tests: FAILED"
        return 1
    fi
    
    # Run robust tests
    if python -m pytest tests/test_robust.py -v --tb=short 2>/dev/null; then
        log_success "Robust tests: PASSED"
    else
        log_error "Robust tests: FAILED"
        return 1
    fi
}

# Check code quality
check_code_quality() {
    log "Checking code quality..."
    
    # Check formatting
    if black --check app/ tests/ 2>/dev/null; then
        log_success "Code formatting: OK"
    else
        log_warning "Code formatting: NEEDS FIXING"
    fi
    
    # Check critical linting errors
    if flake8 app/ --select=E9,F63,F7,F82 --statistics 2>/dev/null; then
        log_success "Critical linting: OK"
    else
        log_warning "Critical linting: ISSUES FOUND"
    fi
}

# Check Docker status
check_docker() {
    log "Checking Docker status..."
    
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null; then
            log_success "Docker daemon: RUNNING"
            
            # Test Docker build
            if docker build -f Dockerfile.simple -t test-build . &> /dev/null; then
                log_success "Docker build: OK"
                docker rmi test-build &> /dev/null
            else
                log_error "Docker build: FAILED"
                return 1
            fi
        else
            log_error "Docker daemon: NOT RUNNING"
            return 1
        fi
    else
        log_warning "Docker not installed"
    fi
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Check disk space
    local disk_usage=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 90 ]; then
        log_success "Disk space: OK (${disk_usage}% used)"
    else
        log_warning "Disk space: LOW (${disk_usage}% used)"
    fi
    
    # Check memory
    if command -v free &> /dev/null; then
        local mem_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
        if [ "$mem_usage" -lt 90 ]; then
            log_success "Memory usage: OK (${mem_usage}%)"
        else
            log_warning "Memory usage: HIGH (${mem_usage}%)"
        fi
    fi
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check Python version
    local python_version=$(python --version 2>&1 | cut -d' ' -f2)
    log_success "Python version: $python_version"
    
    # Check pip packages
    if pip check 2>/dev/null; then
        log_success "Python packages: OK"
    else
        log_warning "Python packages: CONFLICTS DETECTED"
    fi
}

# Performance monitoring
check_performance() {
    log "Checking performance metrics..."
    
    # Test import time
    local import_time=$(time (python -c "from app.main_simple import app" >/dev/null 2>&1) 2>&1 | grep real | awk '{print $2}')
    log_success "Import time: $import_time"
    
    # Test API response time
    local start_time=$(date +%s.%N)
    python -c "
from app.main_simple import app
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.get('/')
" >/dev/null 2>&1
    local end_time=$(date +%s.%N)
    local response_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "N/A")
    log_success "API response time: ${response_time}s"
}

# Generate health report
generate_health_report() {
    log "Generating health report..."
    
    local report_file="/tmp/cicd_health_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "CI/CD Pipeline Health Report"
        echo "Generated: $(date)"
        echo "=========================================="
        echo ""
        
        echo "Application Status:"
        python -c "from app.main_simple import app; print('✅ Simple app: OK')" 2>/dev/null || echo "❌ Simple app: FAILED"
        python -c "import app.models; print('✅ Models: OK')" 2>/dev/null || echo "❌ Models: FAILED"
        python -c "import app.services.mobile_api; print('✅ Services: OK')" 2>/dev/null || echo "❌ Services: FAILED"
        echo ""
        
        echo "Test Status:"
        python -m pytest tests/test_simple_app.py --tb=no -q 2>/dev/null && echo "✅ Simple tests: PASSED" || echo "❌ Simple tests: FAILED"
        python -m pytest tests/test_robust.py --tb=no -q 2>/dev/null && echo "✅ Robust tests: PASSED" || echo "❌ Robust tests: FAILED"
        echo ""
        
        echo "System Information:"
        echo "Python: $(python --version)"
        echo "OS: $(uname -s) $(uname -r)"
        echo "Disk: $(df -h . | awk 'NR==2 {print $5}')"
        echo ""
        
    } > "$report_file"
    
    log_success "Health report generated: $report_file"
}

# Main monitoring function
main() {
    print_header
    
    local failed_checks=0
    local total_checks=0
    
    # Run all checks
    checks=(
        "check_app_health"
        "check_tests"
        "check_code_quality"
        "check_docker"
        "check_system_resources"
        "check_dependencies"
        "check_performance"
    )
    
    for check in "${checks[@]}"; do
        total_checks=$((total_checks + 1))
        if ! $check; then
            failed_checks=$((failed_checks + 1))
        fi
    done
    
    # Summary
    echo ""
    echo -e "${PURPLE}=========================================="
    echo "📊 Monitoring Summary"
    echo "=========================================="
    echo -e "${NC}"
    
    local success_rate=$(( (total_checks - failed_checks) * 100 / total_checks ))
    
    if [ $failed_checks -eq 0 ]; then
        log_success "All checks passed! (${success_rate}% success rate)"
        echo -e "${GREEN}🎉 Pipeline is healthy and ready!${NC}"
    elif [ $failed_checks -le 2 ]; then
        log_warning "Some checks failed (${success_rate}% success rate)"
        echo -e "${YELLOW}⚠️ Pipeline has minor issues but is functional${NC}"
    else
        log_error "Multiple checks failed (${success_rate}% success rate)"
        echo -e "${RED}❌ Pipeline needs attention${NC}"
    fi
    
    # Generate health report
    generate_health_report
    
    # Check GitHub Actions if available
    check_github_actions
    
    echo ""
    echo -e "${CYAN}💡 Tips:${NC}"
    echo "  - Run './scripts/health_check.sh' for detailed diagnostics"
    echo "  - Run './scripts/test_robust_pipeline.sh' for full pipeline test"
    echo "  - Check logs at: $LOG_FILE"
    
    return $failed_checks
}

# Continuous monitoring mode
continuous_monitor() {
    log "Starting continuous monitoring mode (interval: ${HEALTH_CHECK_INTERVAL}s)"
    log "Press Ctrl+C to stop"
    
    while true; do
        echo ""
        echo -e "${CYAN}🔄 Running health check...${NC}"
        main
        echo ""
        echo -e "${BLUE}⏰ Waiting ${HEALTH_CHECK_INTERVAL} seconds until next check...${NC}"
        sleep $HEALTH_CHECK_INTERVAL
    done
}

# Help function
show_help() {
    echo "CI/CD Pipeline Monitor"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --continuous    Run in continuous monitoring mode"
    echo "  -h, --help          Show this help message"
    echo "  -r, --report        Generate health report only"
    echo ""
    echo "Examples:"
    echo "  $0                  # Run single health check"
    echo "  $0 -c               # Run continuous monitoring"
    echo "  $0 -r               # Generate health report"
}

# Parse command line arguments
case "${1:-}" in
    -c|--continuous)
        continuous_monitor
        ;;
    -h|--help)
        show_help
        ;;
    -r|--report)
        generate_health_report
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
