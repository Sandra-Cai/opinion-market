#!/bin/bash

# Final Performance Benchmark Script
# Comprehensive performance testing and validation

set -e

# Configuration
APP_URL="http://localhost:8000"
CONCURRENT_USERS=100
TOTAL_REQUESTS=10000
TEST_DURATION=300  # 5 minutes
WARMUP_TIME=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if hey is installed
    if ! command -v hey &> /dev/null; then
        log_info "Installing hey for performance testing..."
        go install github.com/rakyll/hey@latest
    fi
    
    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed"
        exit 1
    fi
    
    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        log_warning "jq is not installed, JSON parsing will be limited"
    fi
    
    log_success "Dependencies check completed"
}

# Start application
start_application() {
    log_info "Starting application for benchmarking..."
    
    # Check if app is already running
    if curl -s "$APP_URL/health" > /dev/null 2>&1; then
        log_info "Application is already running"
        return 0
    fi
    
    # Start application in background
    log_info "Starting application..."
    python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8000 &
    APP_PID=$!
    
    # Wait for application to start
    log_info "Waiting for application to start..."
    for i in {1..30}; do
        if curl -s "$APP_URL/health" > /dev/null 2>&1; then
            log_success "Application started successfully"
            return 0
        fi
        sleep 1
    done
    
    log_error "Application failed to start"
    exit 1
}

# Warmup
warmup() {
    log_info "Running warmup for $WARMUP_TIME seconds..."
    
    hey -n 100 -c 10 "$APP_URL/health" > /dev/null 2>&1
    sleep $WARMUP_TIME
    
    log_success "Warmup completed"
}

# Health check test
test_health_endpoints() {
    log_info "Testing health endpoints..."
    
    local endpoints=("/health" "/ready" "/metrics")
    local results=()
    
    for endpoint in "${endpoints[@]}"; do
        log_info "Testing endpoint: $endpoint"
        
        local start_time=$(date +%s.%N)
        local response=$(curl -s -w "%{http_code}" "$APP_URL$endpoint")
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        local status_code="${response: -3}"
        
        if [ "$status_code" = "200" ]; then
            log_success "✓ $endpoint: ${duration}s"
            results+=("$endpoint:OK:${duration}")
        else
            log_error "✗ $endpoint: HTTP $status_code"
            results+=("$endpoint:ERROR:$status_code")
        fi
    done
    
    # Save results
    printf '%s\n' "${results[@]}" > health_test_results.txt
    log_success "Health endpoint testing completed"
}

# Load test
load_test() {
    log_info "Running load test..."
    log_info "Configuration: $CONCURRENT_USERS concurrent users, $TOTAL_REQUESTS total requests"
    
    # Test different endpoints
    local endpoints=(
        "/health"
        "/metrics"
        "/api/v1/analytics/performance"
        "/api/v1/analytics/health-score"
    )
    
    for endpoint in "${endpoints[@]}"; do
        log_info "Load testing endpoint: $endpoint"
        
        # Run hey load test
        hey -n $TOTAL_REQUESTS -c $CONCURRENT_USERS "$APP_URL$endpoint" > "load_test_${endpoint//\//_}.txt"
        
        # Extract key metrics
        local summary=$(grep -A 10 "Summary:" "load_test_${endpoint//\//_}.txt" | tail -10)
        echo "$summary" > "summary_${endpoint//\//_}.txt"
        
        log_success "Load test completed for $endpoint"
    done
}

# Stress test
stress_test() {
    log_info "Running stress test..."
    
    # Gradually increase load
    local stress_levels=(50 100 200 500 1000)
    
    for level in "${stress_levels[@]}"; do
        log_info "Stress testing with $level concurrent users..."
        
        hey -n 1000 -c $level -t 30 "$APP_URL/health" > "stress_test_${level}.txt"
        
        # Check if application is still responding
        if ! curl -s "$APP_URL/health" > /dev/null 2>&1; then
            log_error "Application failed under stress at $level concurrent users"
            break
        fi
        
        log_success "Stress test passed at $level concurrent users"
    done
}

# Memory and CPU monitoring
monitor_resources() {
    log_info "Monitoring system resources during test..."
    
    # Start monitoring in background
    (
        while true; do
            echo "$(date),$(ps aux | grep python | grep -v grep | awk '{sum+=$4} END {print sum}'),$(ps aux | grep python | grep -v grep | awk '{sum+=$3} END {print sum}')" >> resource_monitor.csv
            sleep 5
        done
    ) &
    MONITOR_PID=$!
    
    # Run test
    hey -n $TOTAL_REQUESTS -c $CONCURRENT_USERS "$APP_URL/health" > /dev/null
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null || true
    
    log_success "Resource monitoring completed"
}

# API endpoint performance test
test_api_endpoints() {
    log_info "Testing API endpoint performance..."
    
    local api_endpoints=(
        "/api/v1/analytics/performance"
        "/api/v1/analytics/trends"
        "/api/v1/analytics/health-score"
        "/api/v1/analytics/recommendations"
        "/api/v1/security/status"
        "/api/v1/security/headers"
    )
    
    for endpoint in "${api_endpoints[@]}"; do
        log_info "Testing API endpoint: $endpoint"
        
        # Test with different concurrency levels
        for concurrency in 10 50 100; do
            log_info "Testing with $concurrency concurrent requests..."
            
            hey -n 1000 -c $concurrency "$APP_URL$endpoint" > "api_test_${endpoint//\//_}_${concurrency}.txt"
            
            # Extract response time metrics
            local avg_time=$(grep "Average:" "api_test_${endpoint//\//_}_${concurrency}.txt" | awk '{print $2}')
            local max_time=$(grep "Slowest:" "api_test_${endpoint//\//_}_${concurrency}.txt" | awk '{print $2}')
            local rps=$(grep "Requests/sec:" "api_test_${endpoint//\//_}_${concurrency}.txt" | awk '{print $2}')
            
            log_info "Results for $endpoint (c=$concurrency): Avg=${avg_time}s, Max=${max_time}s, RPS=${rps}"
        done
    done
    
    log_success "API endpoint performance testing completed"
}

# Generate performance report
generate_report() {
    log_info "Generating performance report..."
    
    local report_file="performance_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Performance Benchmark Report
Generated: $(date)

## Test Configuration
- Application URL: $APP_URL
- Concurrent Users: $CONCURRENT_USERS
- Total Requests: $TOTAL_REQUESTS
- Test Duration: $TEST_DURATION seconds
- Warmup Time: $WARMUP_TIME seconds

## Health Endpoint Results
EOF

    # Add health endpoint results
    if [ -f "health_test_results.txt" ]; then
        echo "| Endpoint | Status | Response Time |" >> "$report_file"
        echo "|----------|--------|---------------|" >> "$report_file"
        while IFS=: read -r endpoint status time; do
            echo "| $endpoint | $status | ${time}s |" >> "$report_file"
        done < health_test_results.txt
    fi

    cat >> "$report_file" << EOF

## Load Test Results
EOF

    # Add load test results
    for file in load_test_*.txt; do
        if [ -f "$file" ]; then
            local endpoint=$(echo "$file" | sed 's/load_test_//' | sed 's/.txt//' | sed 's/_/\//g')
            echo "### $endpoint" >> "$report_file"
            echo '```' >> "$report_file"
            grep -A 10 "Summary:" "$file" >> "$report_file"
            echo '```' >> "$report_file"
            echo "" >> "$report_file"
        fi
    done

    cat >> "$report_file" << EOF

## Resource Usage
EOF

    # Add resource monitoring results
    if [ -f "resource_monitor.csv" ]; then
        echo "| Timestamp | Memory % | CPU % |" >> "$report_file"
        echo "|-----------|----------|-------|" >> "$report_file"
        tail -20 resource_monitor.csv | while IFS=, read -r timestamp memory cpu; do
            echo "| $timestamp | $memory% | $cpu% |" >> "$report_file"
        done
    fi

    cat >> "$report_file" << EOF

## Recommendations
- Monitor response times under high load
- Consider implementing caching for frequently accessed endpoints
- Optimize database queries if response times are high
- Implement rate limiting to prevent abuse
- Set up alerting for performance degradation

## Files Generated
- Health test results: health_test_results.txt
- Load test results: load_test_*.txt
- Stress test results: stress_test_*.txt
- API test results: api_test_*.txt
- Resource monitoring: resource_monitor.csv
EOF

    log_success "Performance report generated: $report_file"
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Kill application if we started it
    if [ ! -z "$APP_PID" ]; then
        kill $APP_PID 2>/dev/null || true
    fi
    
    # Remove temporary files
    rm -f health_test_results.txt
    rm -f load_test_*.txt
    rm -f stress_test_*.txt
    rm -f api_test_*.txt
    rm -f summary_*.txt
    rm -f resource_monitor.csv
}

# Main execution
main() {
    log_info "Starting comprehensive performance benchmark"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Execute test phases
    check_dependencies
    start_application
    warmup
    test_health_endpoints
    load_test
    stress_test
    monitor_resources
    test_api_endpoints
    generate_report
    
    log_success "Performance benchmark completed successfully!"
}

# Run main function
main "$@"
