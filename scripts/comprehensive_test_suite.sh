#!/bin/bash

# 🧪 Comprehensive Test Suite
# Advanced testing system with multiple test types, reporting, and automation

set -euo pipefail

# Configuration
TEST_LOG="/tmp/comprehensive_test.log"
TEST_RESULTS="/tmp/test_results.json"
TEST_REPORT="/tmp/test_report.md"
COVERAGE_REPORT="/tmp/coverage_report.html"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test categories
UNIT_TESTS="unit"
INTEGRATION_TESTS="integration"
API_TESTS="api"
PERFORMANCE_TESTS="performance"
SECURITY_TESTS="security"
LOAD_TESTS="load"

# Initialize test system
init_test_system() {
    echo -e "${PURPLE}🧪 Initializing Comprehensive Test Suite${NC}"
    
    # Install test dependencies
    install_test_dependencies
    
    # Create test results file
    echo '{"test_suites": [], "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0}}' > "$TEST_RESULTS"
    
    echo -e "${GREEN}✅ Test system initialized${NC}"
}

# Install test dependencies
install_test_dependencies() {
    log_test "Installing test dependencies..."
    
    # Install testing tools
    pip install --quiet pytest pytest-cov pytest-xdist pytest-html pytest-benchmark
    pip install --quiet locust faker factory-boy
    pip install --quiet coverage pytest-mock
    
    log_test_success "Test dependencies installed"
}

# Logging functions
log_test() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$TEST_LOG"
}

log_test_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$TEST_LOG"
}

log_test_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$TEST_LOG"
}

log_test_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$TEST_LOG"
}

# Run unit tests
run_unit_tests() {
    log_test "Running unit tests..."
    
    local start_time=$(date +%s)
    
    # Run unit tests with coverage
    if python -m pytest tests/unit/ -v --tb=short --cov=app --cov-report=html --cov-report=term-missing --junitxml=/tmp/unit_test_results.xml 2>/dev/null; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Parse test results
        local total_tests=$(grep -o 'tests="[0-9]*"' /tmp/unit_test_results.xml | grep -o '[0-9]*' || echo "0")
        local failed_tests=$(grep -o 'failures="[0-9]*"' /tmp/unit_test_results.xml | grep -o '[0-9]*' || echo "0")
        local passed_tests=$((total_tests - failed_tests))
        
        # Update test results
        jq --argjson total "$total_tests" --argjson passed "$passed_tests" --argjson failed "$failed_tests" --argjson duration "$duration" '
            .test_suites += [{
                "name": "unit_tests",
                "total": $total,
                "passed": $passed,
                "failed": $failed,
                "skipped": 0,
                "duration": $duration,
                "status": (if $failed == 0 then "pass" else "fail" end)
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_success "Unit tests completed: $passed_tests/$total_tests passed in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Update test results with failure
        jq --argjson duration "$duration" '
            .test_suites += [{
                "name": "unit_tests",
                "total": 0,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "duration": $duration,
                "status": "fail"
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_error "Unit tests failed"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    log_test "Running integration tests..."
    
    local start_time=$(date +%s)
    
    # Start test services
    start_test_services
    
    # Run integration tests
    if python -m pytest tests/integration/ -v --tb=short --junitxml=/tmp/integration_test_results.xml 2>/dev/null; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Parse test results
        local total_tests=$(grep -o 'tests="[0-9]*"' /tmp/integration_test_results.xml | grep -o '[0-9]*' || echo "0")
        local failed_tests=$(grep -o 'failures="[0-9]*"' /tmp/integration_test_results.xml | grep -o '[0-9]*' || echo "0")
        local passed_tests=$((total_tests - failed_tests))
        
        # Update test results
        jq --argjson total "$total_tests" --argjson passed "$passed_tests" --argjson failed "$failed_tests" --argjson duration "$duration" '
            .test_suites += [{
                "name": "integration_tests",
                "total": $total,
                "passed": $passed,
                "failed": $failed,
                "skipped": 0,
                "duration": $duration,
                "status": (if $failed == 0 then "pass" else "fail" end)
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_success "Integration tests completed: $passed_tests/$total_tests passed in ${duration}s"
        
        # Stop test services
        stop_test_services
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Update test results with failure
        jq --argjson duration "$duration" '
            .test_suites += [{
                "name": "integration_tests",
                "total": 0,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "duration": $duration,
                "status": "fail"
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_error "Integration tests failed"
        
        # Stop test services
        stop_test_services
        return 1
    fi
}

# Start test services
start_test_services() {
    log_test "Starting test services..."
    
    # Start PostgreSQL
    if command -v docker &> /dev/null; then
        docker run -d --name test-postgres -e POSTGRES_PASSWORD=test -e POSTGRES_DB=test_db -p 5433:5432 postgres:15 2>/dev/null || true
        sleep 5
    fi
    
    # Start Redis
    if command -v docker &> /dev/null; then
        docker run -d --name test-redis -p 6380:6379 redis:7 2>/dev/null || true
        sleep 3
    fi
    
    log_test_success "Test services started"
}

# Stop test services
stop_test_services() {
    log_test "Stopping test services..."
    
    # Stop and remove test containers
    docker stop test-postgres test-redis 2>/dev/null || true
    docker rm test-postgres test-redis 2>/dev/null || true
    
    log_test_success "Test services stopped"
}

# Run API tests
run_api_tests() {
    log_test "Running API tests..."
    
    local start_time=$(date +%s)
    
    # Start the application
    python -c "
from app.main_simple import app
import uvicorn
import threading
import time

def run_server():
    uvicorn.run(app, host='127.0.0.1', port=8002, log_level='error')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)  # Wait for server to start
" &
    
    local server_pid=$!
    sleep 5  # Wait for server to be ready
    
    # Run API tests
    if python -m pytest tests/test_simple_app.py -v --tb=short --junitxml=/tmp/api_test_results.xml 2>/dev/null; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Parse test results
        local total_tests=$(grep -o 'tests="[0-9]*"' /tmp/api_test_results.xml | grep -o '[0-9]*' || echo "0")
        local failed_tests=$(grep -o 'failures="[0-9]*"' /tmp/api_test_results.xml | grep -o '[0-9]*' || echo "0")
        local passed_tests=$((total_tests - failed_tests))
        
        # Update test results
        jq --argjson total "$total_tests" --argjson passed "$passed_tests" --argjson failed "$failed_tests" --argjson duration "$duration" '
            .test_suites += [{
                "name": "api_tests",
                "total": $total,
                "passed": $passed,
                "failed": $failed,
                "skipped": 0,
                "duration": $duration,
                "status": (if $failed == 0 then "pass" else "fail" end)
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_success "API tests completed: $passed_tests/$total_tests passed in ${duration}s"
        
        # Kill the test server
        kill $server_pid 2>/dev/null || true
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Update test results with failure
        jq --argjson duration "$duration" '
            .test_suites += [{
                "name": "api_tests",
                "total": 0,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "duration": $duration,
                "status": "fail"
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_error "API tests failed"
        
        # Kill the test server
        kill $server_pid 2>/dev/null || true
        return 1
    fi
}

# Run performance tests
run_performance_tests() {
    log_test "Running performance tests..."
    
    local start_time=$(date +%s)
    
    # Create performance test script
    cat > /tmp/performance_test.py << 'EOF'
import time
import statistics
import requests
from app.main_simple import app
import uvicorn
import threading

def run_server():
    uvicorn.run(app, host='127.0.0.1', port=8003, log_level='error')

# Start server
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

# Test endpoints
endpoints = ["/", "/health", "/api/v1/health", "/metrics"]
results = {}

for endpoint in endpoints:
    response_times = []
    for _ in range(10):
        start = time.time()
        try:
            response = requests.get(f"http://127.0.0.1:8003{endpoint}", timeout=5)
            end = time.time()
            response_times.append(end - start)
        except:
            response_times.append(1.0)  # Failed request
    
    results[endpoint] = {
        "avg": statistics.mean(response_times),
        "min": min(response_times),
        "max": max(response_times),
        "median": statistics.median(response_times)
    }

# Print results
for endpoint, stats in results.items():
    print(f"{endpoint}: avg={stats['avg']:.3f}s, min={stats['min']:.3f}s, max={stats['max']:.3f}s")

# Check if performance is acceptable
avg_response_time = statistics.mean([stats['avg'] for stats in results.values()])
if avg_response_time < 0.5:
    print("PERFORMANCE_TEST: PASS")
    exit(0)
else:
    print("PERFORMANCE_TEST: FAIL")
    exit(1)
EOF
    
    # Run performance test
    if python /tmp/performance_test.py 2>/dev/null; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Update test results
        jq --argjson duration "$duration" '
            .test_suites += [{
                "name": "performance_tests",
                "total": 1,
                "passed": 1,
                "failed": 0,
                "skipped": 0,
                "duration": $duration,
                "status": "pass"
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_success "Performance tests passed in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Update test results with failure
        jq --argjson duration "$duration" '
            .test_suites += [{
                "name": "performance_tests",
                "total": 1,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "duration": $duration,
                "status": "fail"
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_error "Performance tests failed"
        return 1
    fi
}

# Run security tests
run_security_tests() {
    log_test "Running security tests..."
    
    local start_time=$(date +%s)
    
    # Run security scanner
    if ./scripts/security_scanner.sh python 2>/dev/null; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Update test results
        jq --argjson duration "$duration" '
            .test_suites += [{
                "name": "security_tests",
                "total": 1,
                "passed": 1,
                "failed": 0,
                "skipped": 0,
                "duration": $duration,
                "status": "pass"
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_success "Security tests passed in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Update test results with failure
        jq --argjson duration "$duration" '
            .test_suites += [{
                "name": "security_tests",
                "total": 1,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "duration": $duration,
                "status": "fail"
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_error "Security tests failed"
        return 1
    fi
}

# Run load tests
run_load_tests() {
    log_test "Running load tests..."
    
    local start_time=$(date +%s)
    
    # Create load test script
    cat > /tmp/load_test.py << 'EOF'
import time
import threading
import requests
from app.main_simple import app
import uvicorn

def run_server():
    uvicorn.run(app, host='127.0.0.1', port=8004, log_level='error')

# Start server
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

# Load test parameters
num_users = 10
num_requests_per_user = 20
successful_requests = 0
failed_requests = 0

def make_requests():
    global successful_requests, failed_requests
    for _ in range(num_requests_per_user):
        try:
            response = requests.get("http://127.0.0.1:8004/", timeout=5)
            if response.status_code == 200:
                successful_requests += 1
            else:
                failed_requests += 1
        except:
            failed_requests += 1

# Run load test
threads = []
for _ in range(num_users):
    thread = threading.Thread(target=make_requests)
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

total_requests = successful_requests + failed_requests
success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0

print(f"Load test results:")
print(f"Total requests: {total_requests}")
print(f"Successful: {successful_requests}")
print(f"Failed: {failed_requests}")
print(f"Success rate: {success_rate:.2f}%")

if success_rate >= 95:
    print("LOAD_TEST: PASS")
    exit(0)
else:
    print("LOAD_TEST: FAIL")
    exit(1)
EOF
    
    # Run load test
    if python /tmp/load_test.py 2>/dev/null; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Update test results
        jq --argjson duration "$duration" '
            .test_suites += [{
                "name": "load_tests",
                "total": 1,
                "passed": 1,
                "failed": 0,
                "skipped": 0,
                "duration": $duration,
                "status": "pass"
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_success "Load tests passed in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Update test results with failure
        jq --argjson duration "$duration" '
            .test_suites += [{
                "name": "load_tests",
                "total": 1,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "duration": $duration,
                "status": "fail"
            }]
        ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
        
        log_test_error "Load tests failed"
        return 1
    fi
}

# Calculate test summary
calculate_test_summary() {
    local total_tests=0
    local total_passed=0
    local total_failed=0
    local total_skipped=0
    local total_duration=0
    
    # Calculate totals from all test suites
    while IFS= read -r suite; do
        local suite_total=$(echo "$suite" | jq -r '.total')
        local suite_passed=$(echo "$suite" | jq -r '.passed')
        local suite_failed=$(echo "$suite" | jq -r '.failed')
        local suite_skipped=$(echo "$suite" | jq -r '.skipped')
        local suite_duration=$(echo "$suite" | jq -r '.duration')
        
        total_tests=$((total_tests + suite_total))
        total_passed=$((total_passed + suite_passed))
        total_failed=$((total_failed + suite_failed))
        total_skipped=$((total_skipped + suite_skipped))
        total_duration=$((total_duration + suite_duration))
    done < <(jq -c '.test_suites[]' "$TEST_RESULTS")
    
    # Update summary
    jq --argjson total "$total_tests" --argjson passed "$total_passed" --argjson failed "$total_failed" --argjson skipped "$total_skipped" --argjson duration "$total_duration" '
        .summary = {
            "total": $total,
            "passed": $passed,
            "failed": $failed,
            "skipped": $skipped,
            "duration": $duration,
            "success_rate": (if $total > 0 then ($passed * 100 / $total) else 0 end)
        }
    ' "$TEST_RESULTS" > "${TEST_RESULTS}.tmp" && mv "${TEST_RESULTS}.tmp" "$TEST_RESULTS"
    
    echo "$total_tests $total_passed $total_failed $total_skipped $total_duration"
}

# Generate test report
generate_test_report() {
    log_test "Generating test report..."
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local summary=$(calculate_test_summary)
    local total_tests=$(echo $summary | awk '{print $1}')
    local total_passed=$(echo $summary | awk '{print $2}')
    local total_failed=$(echo $summary | awk '{print $3}')
    local total_skipped=$(echo $summary | awk '{print $4}')
    local total_duration=$(echo $summary | awk '{print $5}')
    local success_rate=$(echo "scale=2; $total_passed * 100 / $total_tests" | bc 2>/dev/null || echo "0")
    
    cat > "$TEST_REPORT" << EOF
# 🧪 Comprehensive Test Report

**Generated:** $timestamp  
**Total Tests:** $total_tests  
**Passed:** $total_passed  
**Failed:** $total_failed  
**Skipped:** $total_skipped  
**Success Rate:** ${success_rate}%  
**Total Duration:** ${total_duration}s  

## 📊 Test Suite Results

EOF
    
    # Add test suite results
    jq -r '.test_suites[] | "### \(.name | ascii_upcase | gsub("_"; " "))

- **Total Tests:** \(.total)
- **Passed:** \(.passed)
- **Failed:** \(.failed)
- **Skipped:** \(.skipped)
- **Duration:** \(.duration)s
- **Status:** \(.status | ascii_upcase)

"' "$TEST_RESULTS" >> "$TEST_REPORT"
    
    # Add coverage information if available
    if [[ -f "/tmp/coverage_report.html" ]]; then
        cat >> "$TEST_REPORT" << EOF

## 📈 Code Coverage

Code coverage report is available at: /tmp/coverage_report.html

EOF
    fi
    
    # Add recommendations
    cat >> "$TEST_REPORT" << EOF

## 🎯 Test Recommendations

1. **Increase Coverage:** Aim for >90% code coverage
2. **Add Edge Cases:** Test boundary conditions and error scenarios
3. **Performance Testing:** Set up continuous performance monitoring
4. **Security Testing:** Integrate security scanning into CI/CD
5. **Load Testing:** Test under realistic load conditions
6. **Automated Testing:** Run tests on every commit

## 📈 Test Trends

*Historical test data would be displayed here in a production system.*

## 🔧 Next Steps

1. Review failed tests and fix issues
2. Add missing test cases
3. Improve test coverage
4. Set up continuous testing
5. Monitor test performance trends

EOF
    
    log_test_success "Test report generated: $TEST_REPORT"
}

# Run comprehensive test suite
run_comprehensive_tests() {
    log_test "Starting comprehensive test suite..."
    
    local failed_suites=0
    
    # Run all test suites
    if ! run_unit_tests; then
        failed_suites=$((failed_suites + 1))
    fi
    
    if ! run_integration_tests; then
        failed_suites=$((failed_suites + 1))
    fi
    
    if ! run_api_tests; then
        failed_suites=$((failed_suites + 1))
    fi
    
    if ! run_performance_tests; then
        failed_suites=$((failed_suites + 1))
    fi
    
    if ! run_security_tests; then
        failed_suites=$((failed_suites + 1))
    fi
    
    if ! run_load_tests; then
        failed_suites=$((failed_suites + 1))
    fi
    
    # Generate report
    generate_test_report
    
    # Summary
    local summary=$(calculate_test_summary)
    local total_tests=$(echo $summary | awk '{print $1}')
    local total_passed=$(echo $summary | awk '{print $2}')
    local total_failed=$(echo $summary | awk '{print $3}')
    local success_rate=$(echo "scale=2; $total_passed * 100 / $total_tests" | bc 2>/dev/null || echo "0")
    
    echo ""
    echo -e "${PURPLE}🧪 Comprehensive Test Suite Summary${NC}"
    echo -e "Total tests: $total_tests"
    echo -e "Passed: $total_passed"
    echo -e "Failed: $total_failed"
    echo -e "Success rate: ${success_rate}%"
    echo -e "Failed suites: $failed_suites"
    
    if [[ $failed_suites -eq 0 ]] && (( $(echo "$success_rate >= 90" | bc -l) )); then
        log_test_success "All test suites passed with excellent success rate!"
        echo -e "${GREEN}🎉 Test suite is excellent!${NC}"
    elif [[ $failed_suites -eq 0 ]]; then
        log_test_success "All test suites passed!"
        echo -e "${GREEN}✅ Test suite is good!${NC}"
    else
        log_test_warning "$failed_suites test suite(s) failed"
        echo -e "${YELLOW}⚠️ Some test suites need attention${NC}"
    fi
    
    echo -e "${CYAN}📄 Detailed report: $TEST_REPORT${NC}"
    echo -e "${CYAN}📊 Test data: $TEST_RESULTS${NC}"
}

# Help function
show_help() {
    echo "Comprehensive Test Suite"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  all         Run all test suites"
    echo "  unit        Run unit tests only"
    echo "  integration Run integration tests only"
    echo "  api         Run API tests only"
    echo "  performance Run performance tests only"
    echo "  security    Run security tests only"
    echo "  load        Run load tests only"
    echo "  report      Generate test report"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all      # Run all test suites"
    echo "  $0 unit     # Run unit tests only"
    echo "  $0 report   # Generate test report"
}

# Main function
main() {
    case "${1:-}" in
        all)
            init_test_system
            run_comprehensive_tests
            ;;
        unit)
            init_test_system
            run_unit_tests
            ;;
        integration)
            init_test_system
            run_integration_tests
            ;;
        api)
            init_test_system
            run_api_tests
            ;;
        performance)
            init_test_system
            run_performance_tests
            ;;
        security)
            init_test_system
            run_security_tests
            ;;
        load)
            init_test_system
            run_load_tests
            ;;
        report)
            generate_test_report
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            init_test_system
            run_comprehensive_tests
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
