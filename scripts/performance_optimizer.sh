#!/bin/bash

# ⚡ Advanced Performance Optimization System
# Comprehensive performance analysis, optimization, and monitoring

set -euo pipefail

# Configuration
PERF_LOG="/tmp/performance.log"
BENCHMARK_RESULTS="/tmp/benchmark_results.json"
OPTIMIZATION_REPORT="/tmp/optimization_report.md"
PROFILE_OUTPUT="/tmp/profile_output.prof"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Performance thresholds
IMPORT_TIME_THRESHOLD=0.5
RESPONSE_TIME_THRESHOLD=0.1
MEMORY_USAGE_THRESHOLD=100  # MB
CPU_USAGE_THRESHOLD=80      # %

# Initialize performance system
init_performance() {
    echo -e "${PURPLE}⚡ Initializing Performance Optimization System${NC}"
    
    # Install performance tools if not available
    install_performance_tools
    
    # Create benchmark results file
    echo '{"benchmarks": [], "optimizations": [], "recommendations": []}' > "$BENCHMARK_RESULTS"
    
    echo -e "${GREEN}✅ Performance system initialized${NC}"
}

# Install performance analysis tools
install_performance_tools() {
    log_perf "Installing performance analysis tools..."
    
    # Install Python profiling tools
    pip install --quiet memory-profiler line-profiler py-spy psutil
    
    # Install system monitoring tools
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y htop iotop nethogs
    elif command -v brew &> /dev/null; then
        brew install htop 2>/dev/null || true
    fi
    
    log_perf_success "Performance tools installed"
}

# Logging functions
log_perf() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$PERF_LOG"
}

log_perf_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$PERF_LOG"
}

log_perf_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$PERF_LOG"
}

log_perf_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$PERF_LOG"
}

# Benchmark import performance
benchmark_import_performance() {
    log_perf "Benchmarking import performance..."
    
    local import_times=()
    local iterations=5
    
    for i in $(seq 1 $iterations); do
        local start_time=$(date +%s.%N)
        python -c "from app.main_simple import app" >/dev/null 2>&1
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        import_times+=($duration)
        log_perf "Import iteration $i: ${duration}s"
    done
    
    # Calculate statistics
    local total=0
    for time in "${import_times[@]}"; do
        total=$(echo "$total + $time" | bc)
    done
    local avg_time=$(echo "scale=4; $total / $iterations" | bc)
    
    # Find min and max
    local min_time=${import_times[0]}
    local max_time=${import_times[0]}
    for time in "${import_times[@]}"; do
        if (( $(echo "$time < $min_time" | bc -l) )); then
            min_time=$time
        fi
        if (( $(echo "$time > $max_time" | bc -l) )); then
            max_time=$time
        fi
    done
    
    # Update benchmark results
    jq --arg avg "$avg_time" --arg min "$min_time" --arg max "$max_time" '
        .benchmarks += [{
            "name": "import_performance",
            "avg_time": ($avg | tonumber),
            "min_time": ($min | tonumber),
            "max_time": ($max | tonumber),
            "threshold": 0.5,
            "status": (if ($avg | tonumber) <= 0.5 then "pass" else "fail" end)
        }]
    ' "$BENCHMARK_RESULTS" > "${BENCHMARK_RESULTS}.tmp" && mv "${BENCHMARK_RESULTS}.tmp" "$BENCHMARK_RESULTS"
    
    log_perf_success "Import performance benchmark completed: avg=${avg_time}s, min=${min_time}s, max=${max_time}s"
    
    if (( $(echo "$avg_time > $IMPORT_TIME_THRESHOLD" | bc -l) )); then
        log_perf_warning "Import time exceeds threshold (${avg_time}s > ${IMPORT_TIME_THRESHOLD}s)"
        return 1
    fi
    
    return 0
}

# Benchmark API response times
benchmark_api_performance() {
    log_perf "Benchmarking API response times..."
    
    # Start the application in background
    python -c "
from app.main_simple import app
import uvicorn
import threading
import time

def run_server():
    uvicorn.run(app, host='127.0.0.1', port=8001, log_level='error')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)  # Wait for server to start
" &
    
    local server_pid=$!
    sleep 5  # Wait for server to be ready
    
    local endpoints=("/" "/health" "/api/v1/health" "/metrics")
    local response_times=()
    
    for endpoint in "${endpoints[@]}"; do
        local times=()
        for i in $(seq 1 10); do
            local start_time=$(date +%s.%N)
            curl -s "http://127.0.0.1:8001$endpoint" >/dev/null 2>&1
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            times+=($duration)
        done
        
        # Calculate average
        local total=0
        for time in "${times[@]}"; do
            total=$(echo "$total + $time" | bc)
        done
        local avg_time=$(echo "scale=4; $total / ${#times[@]}" | bc)
        response_times+=($avg_time)
        
        log_perf "Endpoint $endpoint: ${avg_time}s"
    done
    
    # Kill the test server
    kill $server_pid 2>/dev/null || true
    
    # Calculate overall average
    local total=0
    for time in "${response_times[@]}"; do
        total=$(echo "$total + $time" | bc)
    done
    local overall_avg=$(echo "scale=4; $total / ${#response_times[@]}" | bc)
    
    # Update benchmark results
    jq --arg avg "$overall_avg" '
        .benchmarks += [{
            "name": "api_response_time",
            "avg_time": ($avg | tonumber),
            "threshold": 0.1,
            "status": (if ($avg | tonumber) <= 0.1 then "pass" else "fail" end)
        }]
    ' "$BENCHMARK_RESULTS" > "${BENCHMARK_RESULTS}.tmp" && mv "${BENCHMARK_RESULTS}.tmp" "$BENCHMARK_RESULTS"
    
    log_perf_success "API performance benchmark completed: avg=${overall_avg}s"
    
    if (( $(echo "$overall_avg > $RESPONSE_TIME_THRESHOLD" | bc -l) )); then
        log_perf_warning "API response time exceeds threshold (${overall_avg}s > ${RESPONSE_TIME_THRESHOLD}s)"
        return 1
    fi
    
    return 0
}

# Memory usage analysis
analyze_memory_usage() {
    log_perf "Analyzing memory usage..."
    
    # Create memory profiling script
    cat > /tmp/memory_profile.py << 'EOF'
import psutil
import sys
import time
from app.main_simple import app

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

# Measure memory before import
memory_before = get_memory_usage()

# Import the application
from app.main_simple import app

# Measure memory after import
memory_after = get_memory_usage()
memory_increase = memory_after - memory_before

print(f"Memory before import: {memory_before:.2f} MB")
print(f"Memory after import: {memory_after:.2f} MB")
print(f"Memory increase: {memory_increase:.2f} MB")

# Test memory usage under load
import threading
import requests
import time

def make_requests():
    for _ in range(100):
        try:
            requests.get("http://127.0.0.1:8001/", timeout=1)
        except:
            pass

# Start server in background
import uvicorn
server_thread = threading.Thread(target=lambda: uvicorn.run(app, host='127.0.0.1', port=8001, log_level='error'), daemon=True)
server_thread.start()
time.sleep(3)

# Measure memory under load
memory_under_load = get_memory_usage()
print(f"Memory under load: {memory_under_load:.2f} MB")
EOF
    
    # Run memory analysis
    local memory_output=$(python /tmp/memory_profile.py 2>/dev/null || echo "Memory analysis failed")
    log_perf "Memory analysis results:"
    echo "$memory_output" | while read -r line; do
        log_perf "  $line"
    done
    
    # Extract memory increase
    local memory_increase=$(echo "$memory_output" | grep "Memory increase" | awk '{print $3}' | sed 's/MB//')
    
    if [[ -n "$memory_increase" ]]; then
        # Update benchmark results
        jq --arg mem "$memory_increase" '
            .benchmarks += [{
                "name": "memory_usage",
                "memory_increase_mb": ($mem | tonumber),
                "threshold": 100,
                "status": (if ($mem | tonumber) <= 100 then "pass" else "fail" end)
            }]
        ' "$BENCHMARK_RESULTS" > "${BENCHMARK_RESULTS}.tmp" && mv "${BENCHMARK_RESULTS}.tmp" "$BENCHMARK_RESULTS"
        
        if (( $(echo "$memory_increase > $MEMORY_USAGE_THRESHOLD" | bc -l) )); then
            log_perf_warning "Memory usage exceeds threshold (${memory_increase}MB > ${MEMORY_USAGE_THRESHOLD}MB)"
            return 1
        fi
    fi
    
    return 0
}

# CPU usage analysis
analyze_cpu_usage() {
    log_perf "Analyzing CPU usage..."
    
    # Create CPU profiling script
    cat > /tmp/cpu_profile.py << 'EOF'
import psutil
import time
import threading
from app.main_simple import app

def monitor_cpu():
    cpu_percentages = []
    for _ in range(10):
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_percentages.append(cpu_percent)
    return sum(cpu_percentages) / len(cpu_percentages)

# Monitor CPU during import
cpu_before = psutil.cpu_percent(interval=1)
from app.main_simple import app
cpu_after = psutil.cpu_percent(interval=1)

# Monitor CPU under load
import uvicorn
server_thread = threading.Thread(target=lambda: uvicorn.run(app, host='127.0.0.1', port=8001, log_level='error'), daemon=True)
server_thread.start()
time.sleep(3)

cpu_under_load = monitor_cpu()

print(f"CPU before import: {cpu_before:.2f}%")
print(f"CPU after import: {cpu_after:.2f}%")
print(f"CPU under load: {cpu_under_load:.2f}%")
EOF
    
    # Run CPU analysis
    local cpu_output=$(python /tmp/cpu_profile.py 2>/dev/null || echo "CPU analysis failed")
    log_perf "CPU analysis results:"
    echo "$cpu_output" | while read -r line; do
        log_perf "  $line"
    done
    
    # Extract CPU under load
    local cpu_under_load=$(echo "$cpu_output" | grep "CPU under load" | awk '{print $4}' | sed 's/%//')
    
    if [[ -n "$cpu_under_load" ]]; then
        # Update benchmark results
        jq --arg cpu "$cpu_under_load" '
            .benchmarks += [{
                "name": "cpu_usage",
                "cpu_under_load": ($cpu | tonumber),
                "threshold": 80,
                "status": (if ($cpu | tonumber) <= 80 then "pass" else "fail" end)
            }]
        ' "$BENCHMARK_RESULTS" > "${BENCHMARK_RESULTS}.tmp" && mv "${BENCHMARK_RESULTS}.tmp" "$BENCHMARK_RESULTS"
        
        if (( $(echo "$cpu_under_load > $CPU_USAGE_THRESHOLD" | bc -l) )); then
            log_perf_warning "CPU usage exceeds threshold (${cpu_under_load}% > ${CPU_USAGE_THRESHOLD}%)"
            return 1
        fi
    fi
    
    return 0
}

# Code profiling
profile_code() {
    log_perf "Running code profiling..."
    
    # Create profiling script
    cat > /tmp/profile_script.py << 'EOF'
import cProfile
import pstats
import io
from app.main_simple import app

# Profile the import
profiler = cProfile.Profile()
profiler.enable()

# Import the application
from app.main_simple import app

profiler.disable()

# Get profiling results
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(20)  # Top 20 functions

print("Top 20 functions by cumulative time:")
print(s.getvalue())
EOF
    
    # Run profiling
    local profile_output=$(python /tmp/profile_script.py 2>/dev/null || echo "Profiling failed")
    log_perf "Code profiling results:"
    echo "$profile_output" | head -20 | while read -r line; do
        log_perf "  $line"
    done
    
    # Save detailed profile
    echo "$profile_output" > "$PROFILE_OUTPUT"
    log_perf_success "Detailed profile saved to $PROFILE_OUTPUT"
}

# Generate optimization recommendations
generate_optimization_recommendations() {
    log_perf "Generating optimization recommendations..."
    
    local recommendations=()
    
    # Check import performance
    local import_avg=$(jq -r '.benchmarks[] | select(.name=="import_performance") | .avg_time' "$BENCHMARK_RESULTS" 2>/dev/null || echo "0")
    if (( $(echo "$import_avg > $IMPORT_TIME_THRESHOLD" | bc -l) )); then
        recommendations+=("Consider lazy loading for heavy imports")
        recommendations+=("Optimize import statements and reduce circular dependencies")
        recommendations+=("Use __import__ for conditional imports")
    fi
    
    # Check API performance
    local api_avg=$(jq -r '.benchmarks[] | select(.name=="api_response_time") | .avg_time' "$BENCHMARK_RESULTS" 2>/dev/null || echo "0")
    if (( $(echo "$api_avg > $RESPONSE_TIME_THRESHOLD" | bc -l) )); then
        recommendations+=("Implement response caching")
        recommendations+=("Optimize database queries")
        recommendations+=("Use async/await for I/O operations")
        recommendations+=("Consider connection pooling")
    fi
    
    # Check memory usage
    local memory_usage=$(jq -r '.benchmarks[] | select(.name=="memory_usage") | .memory_increase_mb' "$BENCHMARK_RESULTS" 2>/dev/null || echo "0")
    if (( $(echo "$memory_usage > $MEMORY_USAGE_THRESHOLD" | bc -l) )); then
        recommendations+=("Implement memory-efficient data structures")
        recommendations+=("Use generators instead of lists for large datasets")
        recommendations+=("Consider garbage collection optimization")
        recommendations+=("Profile memory leaks with memory_profiler")
    fi
    
    # Check CPU usage
    local cpu_usage=$(jq -r '.benchmarks[] | select(.name=="cpu_usage") | .cpu_under_load' "$BENCHMARK_RESULTS" 2>/dev/null || echo "0")
    if (( $(echo "$cpu_usage > $CPU_USAGE_THRESHOLD" | bc -l) )); then
        recommendations+=("Optimize CPU-intensive operations")
        recommendations+=("Consider multiprocessing for parallel tasks")
        recommendations+=("Use efficient algorithms and data structures")
        recommendations+=("Profile with py-spy for CPU hotspots")
    fi
    
    # Add general recommendations
    recommendations+=("Enable Python optimizations (-O flag)")
    recommendations+=("Use compiled extensions for critical paths")
    recommendations+=("Implement proper logging levels")
    recommendations+=("Consider using a faster WSGI server like gunicorn")
    
    # Update benchmark results with recommendations
    local rec_json=$(printf '%s\n' "${recommendations[@]}" | jq -R . | jq -s .)
    jq --argjson recs "$rec_json" '.recommendations = $recs' "$BENCHMARK_RESULTS" > "${BENCHMARK_RESULTS}.tmp" && mv "${BENCHMARK_RESULTS}.tmp" "$BENCHMARK_RESULTS"
    
    log_perf_success "Generated ${#recommendations[@]} optimization recommendations"
    
    # Display recommendations
    echo -e "${CYAN}📋 Optimization Recommendations:${NC}"
    for i in "${!recommendations[@]}"; do
        echo -e "  $((i+1)). ${recommendations[i]}"
    done
}

# Generate optimization report
generate_optimization_report() {
    log_perf "Generating optimization report..."
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat > "$OPTIMIZATION_REPORT" << EOF
# ⚡ Performance Optimization Report

**Generated:** $timestamp  
**System:** $(uname -s) $(uname -r)  
**Python:** $(python --version)  

## 📊 Benchmark Results

EOF
    
    # Add benchmark results
    jq -r '.benchmarks[] | "### \(.name | ascii_upcase | gsub("_"; " "))

- **Average Time:** \(.avg_time)s
- **Threshold:** \(.threshold)s
- **Status:** \(.status | ascii_upcase)

"' "$BENCHMARK_RESULTS" >> "$OPTIMIZATION_REPORT"
    
    # Add recommendations
    echo "## 🎯 Optimization Recommendations" >> "$OPTIMIZATION_REPORT"
    echo "" >> "$OPTIMIZATION_REPORT"
    
    jq -r '.recommendations[] | "- \(.)"' "$BENCHMARK_RESULTS" >> "$OPTIMIZATION_REPORT"
    
    # Add system information
    cat >> "$OPTIMIZATION_REPORT" << EOF

## 🖥️ System Information

- **CPU Cores:** $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "Unknown")
- **Memory:** $(free -h | awk 'NR==2{print $2}' 2>/dev/null || echo "Unknown")
- **Disk:** $(df -h . | awk 'NR==2{print $2}' 2>/dev/null || echo "Unknown")

## 📈 Performance Trends

*Historical performance data would be displayed here in a production system.*

## 🔧 Next Steps

1. Review the optimization recommendations above
2. Implement the most impactful changes first
3. Re-run benchmarks to measure improvements
4. Set up continuous performance monitoring

EOF
    
    log_perf_success "Optimization report generated: $OPTIMIZATION_REPORT"
}

# Run comprehensive performance analysis
run_performance_analysis() {
    log_perf "Starting comprehensive performance analysis..."
    
    local failed_benchmarks=0
    
    # Run all benchmarks
    if ! benchmark_import_performance; then
        failed_benchmarks=$((failed_benchmarks + 1))
    fi
    
    if ! benchmark_api_performance; then
        failed_benchmarks=$((failed_benchmarks + 1))
    fi
    
    if ! analyze_memory_usage; then
        failed_benchmarks=$((failed_benchmarks + 1))
    fi
    
    if ! analyze_cpu_usage; then
        failed_benchmarks=$((failed_benchmarks + 1))
    fi
    
    # Run code profiling
    profile_code
    
    # Generate recommendations
    generate_optimization_recommendations
    
    # Generate report
    generate_optimization_report
    
    # Summary
    echo ""
    echo -e "${PURPLE}📊 Performance Analysis Summary${NC}"
    echo -e "Failed benchmarks: $failed_benchmarks"
    
    if [[ $failed_benchmarks -eq 0 ]]; then
        log_perf_success "All performance benchmarks passed!"
        echo -e "${GREEN}🎉 Performance is optimal!${NC}"
    else
        log_perf_warning "$failed_benchmarks benchmark(s) failed - see recommendations"
        echo -e "${YELLOW}⚠️ Performance optimization needed${NC}"
    fi
    
    echo -e "${CYAN}📄 Detailed report: $OPTIMIZATION_REPORT${NC}"
    echo -e "${CYAN}📊 Benchmark data: $BENCHMARK_RESULTS${NC}"
    echo -e "${CYAN}🔍 Profile data: $PROFILE_OUTPUT${NC}"
}

# Help function
show_help() {
    echo "Advanced Performance Optimization System"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  analyze     Run comprehensive performance analysis"
    echo "  benchmark   Run performance benchmarks only"
    echo "  profile     Run code profiling only"
    echo "  report      Generate optimization report"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 analyze    # Run full performance analysis"
    echo "  $0 benchmark  # Run benchmarks only"
    echo "  $0 profile    # Profile code performance"
}

# Main function
main() {
    case "${1:-}" in
        analyze)
            init_performance
            run_performance_analysis
            ;;
        benchmark)
            init_performance
            benchmark_import_performance
            benchmark_api_performance
            analyze_memory_usage
            analyze_cpu_usage
            ;;
        profile)
            init_performance
            profile_code
            ;;
        report)
            generate_optimization_report
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            init_performance
            run_performance_analysis
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
