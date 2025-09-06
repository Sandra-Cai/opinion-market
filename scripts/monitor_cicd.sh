#!/bin/bash

# 📊 CI/CD Monitoring Dashboard Script
# Real-time monitoring of CI/CD pipeline health and performance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration
REFRESH_INTERVAL=30
LOG_FILE="/tmp/cicd_monitor.log"
METRICS_FILE="/tmp/cicd_metrics.json"

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
    echo "📊 CI/CD Monitoring Dashboard"
    echo "=========================================="
    echo -e "${NC}"
}

# Metrics collection
collect_metrics() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # System metrics
    local cpu_usage=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
    local memory_usage=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    local disk_usage=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
    
    # Docker metrics
    local docker_containers=$(docker ps -q | wc -l)
    local docker_images=$(docker images -q | wc -l)
    
    # Git metrics
    local git_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    local git_commits=$(git rev-list --count HEAD 2>/dev/null || echo "0")
    local git_status=$(git status --porcelain | wc -l)
    
    # Test metrics
    local test_files=$(find tests/ -name "*.py" | wc -l)
    local test_functions=$(grep -r "def test_" tests/ | wc -l)
    
    # Code metrics
    local python_files=$(find app/ -name "*.py" | wc -l)
    local total_lines=$(find app/ tests/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')
    
    # Create metrics JSON
    cat > "$METRICS_FILE" << EOF
{
    "timestamp": "$timestamp",
    "system": {
        "cpu_usage": "$cpu_usage",
        "memory_usage": "$memory_usage",
        "disk_usage": "$disk_usage"
    },
    "docker": {
        "containers": "$docker_containers",
        "images": "$docker_images"
    },
    "git": {
        "branch": "$git_branch",
        "commits": "$git_commits",
        "uncommitted_changes": "$git_status"
    },
    "tests": {
        "test_files": "$test_files",
        "test_functions": "$test_functions"
    },
    "code": {
        "python_files": "$python_files",
        "total_lines": "$total_lines"
    }
}
EOF
}

# Display metrics
display_metrics() {
    if [ ! -f "$METRICS_FILE" ]; then
        log_error "No metrics file found"
        return 1
    fi
    
    local timestamp=$(jq -r '.timestamp' "$METRICS_FILE")
    local cpu_usage=$(jq -r '.system.cpu_usage' "$METRICS_FILE")
    local memory_usage=$(jq -r '.system.memory_usage' "$METRICS_FILE")
    local disk_usage=$(jq -r '.system.disk_usage' "$METRICS_FILE")
    local docker_containers=$(jq -r '.docker.containers' "$METRICS_FILE")
    local docker_images=$(jq -r '.docker.images' "$METRICS_FILE")
    local git_branch=$(jq -r '.git.branch' "$METRICS_FILE")
    local git_commits=$(jq -r '.git.commits' "$METRICS_FILE")
    local git_status=$(jq -r '.git.uncommitted_changes' "$METRICS_FILE")
    local test_files=$(jq -r '.tests.test_files' "$METRICS_FILE")
    local test_functions=$(jq -r '.tests.test_functions' "$METRICS_FILE")
    local python_files=$(jq -r '.code.python_files' "$METRICS_FILE")
    local total_lines=$(jq -r '.code.total_lines' "$METRICS_FILE")
    
    # Clear screen and display header
    clear
    log_header
    echo -e "${WHITE}Last Updated: $timestamp${NC}"
    echo ""
    
    # System metrics
    echo -e "${CYAN}🖥️  System Metrics${NC}"
    echo "├─ CPU Usage: $cpu_usage%"
    echo "├─ Memory Usage: $memory_usage%"
    echo "└─ Disk Usage: $disk_usage%"
    echo ""
    
    # Docker metrics
    echo -e "${CYAN}🐳 Docker Metrics${NC}"
    echo "├─ Running Containers: $docker_containers"
    echo "└─ Available Images: $docker_images"
    echo ""
    
    # Git metrics
    echo -e "${CYAN}📝 Git Metrics${NC}"
    echo "├─ Current Branch: $git_branch"
    echo "├─ Total Commits: $git_commits"
    echo "└─ Uncommitted Changes: $git_status"
    echo ""
    
    # Code metrics
    echo -e "${CYAN}💻 Code Metrics${NC}"
    echo "├─ Python Files: $python_files"
    echo "├─ Total Lines: $total_lines"
    echo "├─ Test Files: $test_files"
    echo "└─ Test Functions: $test_functions"
    echo ""
    
    # Health status
    echo -e "${CYAN}🏥 Health Status${NC}"
    
    # Check system health
    if [ "$cpu_usage" -lt 80 ] && [ "$disk_usage" -lt 90 ]; then
        echo "├─ System: $(log_success "Healthy")"
    else
        echo "├─ System: $(log_warning "Warning")"
    fi
    
    # Check Docker health
    if [ "$docker_containers" -gt 0 ]; then
        echo "├─ Docker: $(log_success "Active")"
    else
        echo "├─ Docker: $(log_warning "No containers")"
    fi
    
    # Check Git health
    if [ "$git_status" -eq 0 ]; then
        echo "├─ Git: $(log_success "Clean")"
    else
        echo "├─ Git: $(log_warning "Uncommitted changes")"
    fi
    
    # Check test health
    if [ "$test_functions" -gt 10 ]; then
        echo "└─ Tests: $(log_success "Good coverage")"
    else
        echo "└─ Tests: $(log_warning "Low coverage")"
    fi
    
    echo ""
    echo -e "${WHITE}Press Ctrl+C to exit monitoring${NC}"
}

# Check GitHub Actions status (if gh CLI is available)
check_github_actions() {
    if command -v gh &> /dev/null; then
        echo -e "${CYAN}🔄 GitHub Actions Status${NC}"
        local runs=$(gh run list --limit 5 --json status,conclusion,createdAt,headBranch)
        echo "$runs" | jq -r '.[] | "├─ \(.headBranch): \(.status) (\(.conclusion // "in_progress")) - \(.createdAt)"'
        echo ""
    else
        echo -e "${YELLOW}⚠️  GitHub CLI not available for Actions status${NC}"
        echo ""
    fi
}

# Monitor mode
monitor_mode() {
    log_info "Starting CI/CD monitoring dashboard..."
    log_info "Refresh interval: ${REFRESH_INTERVAL} seconds"
    echo ""
    
    while true; do
        collect_metrics
        display_metrics
        check_github_actions
        
        sleep $REFRESH_INTERVAL
    done
}

# Single run mode
single_run() {
    log_info "Running single CI/CD metrics collection..."
    collect_metrics
    display_metrics
    check_github_actions
}

# Help function
show_help() {
    echo "CI/CD Monitoring Dashboard"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --monitor    Run in continuous monitoring mode (default)"
    echo "  -s, --single     Run single metrics collection"
    echo "  -i, --interval   Set refresh interval in seconds (default: 30)"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start monitoring dashboard"
    echo "  $0 --single           # Run single metrics collection"
    echo "  $0 --interval 60      # Monitor with 60-second refresh"
}

# Main function
main() {
    local mode="monitor"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--monitor)
                mode="monitor"
                shift
                ;;
            -s|--single)
                mode="single"
                shift
                ;;
            -i|--interval)
                REFRESH_INTERVAL="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check dependencies
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed. Please install jq."
        exit 1
    fi
    
    # Create log file
    touch "$LOG_FILE"
    
    # Run in specified mode
    case $mode in
        "monitor")
            monitor_mode
            ;;
        "single")
            single_run
            ;;
        *)
            log_error "Invalid mode: $mode"
            exit 1
            ;;
    esac
}

# Cleanup function
cleanup() {
    log_info "Cleaning up monitoring artifacts..."
    rm -f "$LOG_FILE" "$METRICS_FILE"
}

# Set up trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"
