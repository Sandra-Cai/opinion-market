#!/bin/bash

# 🚀 Advanced CI/CD Pipeline Monitoring & Alerting System
# Comprehensive monitoring with real-time alerts and automated responses

set -euo pipefail

# Configuration
LOG_FILE="/tmp/advanced_monitor.log"
ALERT_LOG="/tmp/alert_log.log"
METRICS_FILE="/tmp/metrics.json"
CONFIG_FILE="monitoring-config.yaml"
HEALTH_CHECK_INTERVAL=30
ALERT_COOLDOWN=300  # 5 minutes
MAX_RETRIES=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Alert levels
CRITICAL="critical"
WARNING="warning"
INFO="info"

# Initialize monitoring
init_monitoring() {
    echo -e "${PURPLE}🚀 Initializing Advanced Monitoring System${NC}"
    
    # Create monitoring config if it doesn't exist
    if [[ ! -f "$CONFIG_FILE" ]]; then
        create_default_config
    fi
    
    # Initialize metrics
    init_metrics
    
    # Set up signal handlers
    trap cleanup EXIT INT TERM
    
    echo -e "${GREEN}✅ Monitoring system initialized${NC}"
}

# Create default monitoring configuration
create_default_config() {
    cat > "$CONFIG_FILE" << 'EOF'
monitoring:
  metrics:
    - name: "response_time"
      threshold: 1.0
      unit: "seconds"
      critical_threshold: 2.0
    - name: "error_rate"
      threshold: 0.05
      unit: "percentage"
      critical_threshold: 0.1
    - name: "cpu_usage"
      threshold: 80
      unit: "percentage"
      critical_threshold: 90
    - name: "memory_usage"
      threshold: 85
      unit: "percentage"
      critical_threshold: 95
    - name: "disk_usage"
      threshold: 90
      unit: "percentage"
      critical_threshold: 95
    - name: "import_time"
      threshold: 0.5
      unit: "seconds"
      critical_threshold: 1.0
  
  alerts:
    - name: "high_response_time"
      condition: "response_time > 1.0"
      severity: "warning"
      action: "restart_service"
    - name: "critical_response_time"
      condition: "response_time > 2.0"
      severity: "critical"
      action: "emergency_restart"
    - name: "high_error_rate"
      condition: "error_rate > 0.05"
      severity: "warning"
      action: "check_logs"
    - name: "critical_error_rate"
      condition: "error_rate > 0.1"
      severity: "critical"
      action: "rollback_deployment"
    - name: "resource_exhaustion"
      condition: "cpu_usage > 80 OR memory_usage > 85"
      severity: "warning"
      action: "scale_resources"
    - name: "critical_resource_exhaustion"
      condition: "cpu_usage > 90 OR memory_usage > 95"
      severity: "critical"
      action: "emergency_scale"
  
  notifications:
    - type: "slack"
      webhook: "${SLACK_WEBHOOK_URL}"
      channels: ["#alerts", "#devops"]
    - type: "email"
      recipients: ["admin@company.com", "devops@company.com"]
    - type: "pagerduty"
      service_key: "${PAGERDUTY_SERVICE_KEY}"
  
  actions:
    - name: "restart_service"
      command: "docker restart opinion-market"
      timeout: 30
    - name: "emergency_restart"
      command: "docker restart opinion-market && sleep 10 && docker logs opinion-market"
      timeout: 60
    - name: "check_logs"
      command: "docker logs --tail 100 opinion-market"
      timeout: 10
    - name: "rollback_deployment"
      command: "kubectl rollout undo deployment/opinion-market"
      timeout: 120
    - name: "scale_resources"
      command: "kubectl scale deployment opinion-market --replicas=3"
      timeout: 60
    - name: "emergency_scale"
      command: "kubectl scale deployment opinion-market --replicas=5"
      timeout: 60
EOF
    
    echo -e "${GREEN}✅ Default monitoring configuration created${NC}"
}

# Initialize metrics collection
init_metrics() {
    cat > "$METRICS_FILE" << 'EOF'
{
  "timestamp": null,
  "metrics": {
    "response_time": 0,
    "error_rate": 0,
    "cpu_usage": 0,
    "memory_usage": 0,
    "disk_usage": 0,
    "import_time": 0
  },
  "alerts": [],
  "health_score": 100
}
EOF
}

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_alert() {
    echo -e "${RED}[ALERT]${NC} $1" | tee -a "$ALERT_LOG"
}

# Collect system metrics
collect_metrics() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Response time
    local response_time=0
    if command -v curl &> /dev/null; then
        response_time=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8000/ 2>/dev/null || echo "0")
    fi
    
    # Error rate (simplified - would be more complex in production)
    local error_rate=0
    if [[ -f "/tmp/error_count" ]]; then
        local error_count=$(cat /tmp/error_count)
        local total_requests=$(cat /tmp/total_requests 2>/dev/null || echo "100")
        error_rate=$(echo "scale=4; $error_count / $total_requests" | bc 2>/dev/null || echo "0")
    fi
    
    # CPU usage
    local cpu_usage=0
    if command -v top &> /dev/null; then
        cpu_usage=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//' 2>/dev/null || echo "0")
    fi
    
    # Memory usage
    local memory_usage=0
    if command -v vm_stat &> /dev/null; then
        # macOS memory calculation
        memory_usage=$(vm_stat | awk '/Pages free/ {free=$3} /Pages active/ {active=$3} /Pages inactive/ {inactive=$3} /Pages speculative/ {spec=$3} /Pages wired down/ {wired=$4} END {total=free+active+inactive+spec+wired; used=active+inactive+wired; printf "%.0f", used*100/total}' 2>/dev/null || echo "0")
    elif command -v free &> /dev/null; then
        # Linux memory calculation
        memory_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}' 2>/dev/null || echo "0")
    fi
    
    # Disk usage
    local disk_usage=0
    if command -v df &> /dev/null; then
        disk_usage=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//' 2>/dev/null || echo "0")
    fi
    
    # Import time
    local import_time=0
    if command -v time &> /dev/null; then
        import_time=$(time (python -c "from app.main_simple import app" >/dev/null 2>&1) 2>&1 | grep real | awk '{print $2}' | sed 's/[ms]//g' 2>/dev/null || echo "0")
    fi
    
    # Update metrics file
    cat > "$METRICS_FILE" << EOF
{
  "timestamp": "$timestamp",
  "metrics": {
    "response_time": $response_time,
    "error_rate": $error_rate,
    "cpu_usage": $cpu_usage,
    "memory_usage": $memory_usage,
    "disk_usage": $disk_usage,
    "import_time": $import_time
  },
  "alerts": [],
  "health_score": 100
}
EOF
    
    log "Metrics collected: RT=${response_time}s, ER=${error_rate}, CPU=${cpu_usage}%, MEM=${memory_usage}%, DISK=${disk_usage}%"
}

# Check alert conditions
check_alerts() {
    local alerts_triggered=0
    
    # Read current metrics
    local response_time=$(jq -r '.metrics.response_time' "$METRICS_FILE")
    local error_rate=$(jq -r '.metrics.error_rate' "$METRICS_FILE")
    local cpu_usage=$(jq -r '.metrics.cpu_usage' "$METRICS_FILE")
    local memory_usage=$(jq -r '.metrics.memory_usage' "$METRICS_FILE")
    local disk_usage=$(jq -r '.metrics.disk_usage' "$METRICS_FILE")
    local import_time=$(jq -r '.metrics.import_time' "$METRICS_FILE")
    
    # Check response time alerts
    if (( $(echo "$response_time > 2.0" | bc -l) )); then
        trigger_alert "critical_response_time" "$CRITICAL" "Response time critical: ${response_time}s"
        alerts_triggered=$((alerts_triggered + 1))
    elif (( $(echo "$response_time > 1.0" | bc -l) )); then
        trigger_alert "high_response_time" "$WARNING" "Response time high: ${response_time}s"
        alerts_triggered=$((alerts_triggered + 1))
    fi
    
    # Check error rate alerts
    if (( $(echo "$error_rate > 0.1" | bc -l) )); then
        trigger_alert "critical_error_rate" "$CRITICAL" "Error rate critical: ${error_rate}"
        alerts_triggered=$((alerts_triggered + 1))
    elif (( $(echo "$error_rate > 0.05" | bc -l) )); then
        trigger_alert "high_error_rate" "$WARNING" "Error rate high: ${error_rate}"
        alerts_triggered=$((alerts_triggered + 1))
    fi
    
    # Check resource alerts
    if (( $(echo "$cpu_usage > 90" | bc -l) )) || (( $(echo "$memory_usage > 95" | bc -l) )); then
        trigger_alert "critical_resource_exhaustion" "$CRITICAL" "Resource exhaustion: CPU=${cpu_usage}%, MEM=${memory_usage}%"
        alerts_triggered=$((alerts_triggered + 1))
    elif (( $(echo "$cpu_usage > 80" | bc -l) )) || (( $(echo "$memory_usage > 85" | bc -l) )); then
        trigger_alert "resource_exhaustion" "$WARNING" "High resource usage: CPU=${cpu_usage}%, MEM=${memory_usage}%"
        alerts_triggered=$((alerts_triggered + 1))
    fi
    
    # Check disk usage
    if (( $(echo "$disk_usage > 95" | bc -l) )); then
        trigger_alert "critical_disk_usage" "$CRITICAL" "Disk usage critical: ${disk_usage}%"
        alerts_triggered=$((alerts_triggered + 1))
    elif (( $(echo "$disk_usage > 90" | bc -l) )); then
        trigger_alert "high_disk_usage" "$WARNING" "Disk usage high: ${disk_usage}%"
        alerts_triggered=$((alerts_triggered + 1))
    fi
    
    # Check import time
    if (( $(echo "$import_time > 1.0" | bc -l) )); then
        trigger_alert "slow_import" "$WARNING" "Import time slow: ${import_time}s"
        alerts_triggered=$((alerts_triggered + 1))
    fi
    
    return $alerts_triggered
}

# Trigger alert and execute action
trigger_alert() {
    local alert_name="$1"
    local severity="$2"
    local message="$3"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Check cooldown
    if [[ -f "/tmp/alert_${alert_name}_last" ]]; then
        local last_alert=$(cat "/tmp/alert_${alert_name}_last")
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_alert))
        
        if [[ $time_diff -lt $ALERT_COOLDOWN ]]; then
            log "Alert $alert_name in cooldown period"
            return
        fi
    fi
    
    # Log alert
    log_alert "[$severity] $alert_name: $message"
    
    # Record alert timestamp
    echo $(date +%s) > "/tmp/alert_${alert_name}_last"
    
    # Send notifications
    send_notifications "$severity" "$alert_name" "$message"
    
    # Execute action based on severity
    case "$severity" in
        "$CRITICAL")
            execute_critical_action "$alert_name"
            ;;
        "$WARNING")
            execute_warning_action "$alert_name"
            ;;
    esac
}

# Send notifications
send_notifications() {
    local severity="$1"
    local alert_name="$2"
    local message="$3"
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color="good"
        case "$severity" in
            "$CRITICAL") color="danger" ;;
            "$WARNING") color="warning" ;;
        esac
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 $severity Alert: $alert_name\", \"attachments\":[{\"color\":\"$color\", \"text\":\"$message\"}]}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null || true
    fi
    
    # Email notification (simplified)
    if [[ -n "${ALERT_EMAIL:-}" ]]; then
        echo "Subject: [$severity] $alert_name - $message" | sendmail "$ALERT_EMAIL" 2>/dev/null || true
    fi
    
    log "Notifications sent for $alert_name"
}

# Execute critical actions
execute_critical_action() {
    local alert_name="$1"
    
    case "$alert_name" in
        "critical_response_time")
            log "Executing emergency restart..."
            docker restart opinion-market 2>/dev/null || true
            ;;
        "critical_error_rate")
            log "Executing rollback deployment..."
            kubectl rollout undo deployment/opinion-market 2>/dev/null || true
            ;;
        "critical_resource_exhaustion")
            log "Executing emergency scaling..."
            kubectl scale deployment opinion-market --replicas=5 2>/dev/null || true
            ;;
        "critical_disk_usage")
            log "Executing disk cleanup..."
            docker system prune -f 2>/dev/null || true
            ;;
    esac
}

# Execute warning actions
execute_warning_action() {
    local alert_name="$1"
    
    case "$alert_name" in
        "high_response_time")
            log "Checking service logs..."
            docker logs --tail 50 opinion-market 2>/dev/null || true
            ;;
        "high_error_rate")
            log "Checking error logs..."
            docker logs --tail 100 opinion-market 2>/dev/null | grep -i error || true
            ;;
        "resource_exhaustion")
            log "Scaling resources..."
            kubectl scale deployment opinion-market --replicas=3 2>/dev/null || true
            ;;
        "high_disk_usage")
            log "Cleaning up temporary files..."
            find /tmp -name "*.log" -mtime +1 -delete 2>/dev/null || true
            ;;
    esac
}

# Calculate health score
calculate_health_score() {
    local score=100
    
    # Read metrics
    local response_time=$(jq -r '.metrics.response_time' "$METRICS_FILE")
    local error_rate=$(jq -r '.metrics.error_rate' "$METRICS_FILE")
    local cpu_usage=$(jq -r '.metrics.cpu_usage' "$METRICS_FILE")
    local memory_usage=$(jq -r '.metrics.memory_usage' "$METRICS_FILE")
    local disk_usage=$(jq -r '.metrics.disk_usage' "$METRICS_FILE")
    
    # Deduct points for issues
    if (( $(echo "$response_time > 2.0" | bc -l) )); then
        score=$((score - 30))
    elif (( $(echo "$response_time > 1.0" | bc -l) )); then
        score=$((score - 15))
    fi
    
    if (( $(echo "$error_rate > 0.1" | bc -l) )); then
        score=$((score - 25))
    elif (( $(echo "$error_rate > 0.05" | bc -l) )); then
        score=$((score - 10))
    fi
    
    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        score=$((score - 20))
    elif (( $(echo "$cpu_usage > 80" | bc -l) )); then
        score=$((score - 10))
    fi
    
    if (( $(echo "$memory_usage > 95" | bc -l) )); then
        score=$((score - 20))
    elif (( $(echo "$memory_usage > 85" | bc -l) )); then
        score=$((score - 10))
    fi
    
    if (( $(echo "$disk_usage > 95" | bc -l) )); then
        score=$((score - 15))
    elif (( $(echo "$disk_usage > 90" | bc -l) )); then
        score=$((score - 5))
    fi
    
    # Ensure score doesn't go below 0
    if [[ $score -lt 0 ]]; then
        score=0
    fi
    
    # Update metrics file
    jq ".health_score = $score" "$METRICS_FILE" > "${METRICS_FILE}.tmp" && mv "${METRICS_FILE}.tmp" "$METRICS_FILE"
    
    echo $score
}

# Generate monitoring dashboard
generate_dashboard() {
    local health_score=$(calculate_health_score)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    clear
    echo -e "${PURPLE}"
    echo "=========================================="
    echo "🚀 Advanced CI/CD Monitoring Dashboard"
    echo "=========================================="
    echo -e "${NC}"
    echo -e "${CYAN}Last Updated: $timestamp${NC}"
    echo ""
    
    # Health score
    local health_color="$GREEN"
    if [[ $health_score -lt 70 ]]; then
        health_color="$RED"
    elif [[ $health_score -lt 85 ]]; then
        health_color="$YELLOW"
    fi
    echo -e "🏥 Health Score: ${health_color}$health_score%${NC}"
    echo ""
    
    # Metrics
    if [[ -f "$METRICS_FILE" ]]; then
        local response_time=$(jq -r '.metrics.response_time' "$METRICS_FILE")
        local error_rate=$(jq -r '.metrics.error_rate' "$METRICS_FILE")
        local cpu_usage=$(jq -r '.metrics.cpu_usage' "$METRICS_FILE")
        local memory_usage=$(jq -r '.metrics.memory_usage' "$METRICS_FILE")
        local disk_usage=$(jq -r '.metrics.disk_usage' "$METRICS_FILE")
        local import_time=$(jq -r '.metrics.import_time' "$METRICS_FILE")
        
        echo -e "${BLUE}📊 Current Metrics:${NC}"
        echo -e "  Response Time: ${response_time}s"
        echo -e "  Error Rate: ${error_rate}"
        echo -e "  CPU Usage: ${cpu_usage}%"
        echo -e "  Memory Usage: ${memory_usage}%"
        echo -e "  Disk Usage: ${disk_usage}%"
        echo -e "  Import Time: ${import_time}s"
        echo ""
    fi
    
    # Recent alerts
    if [[ -f "$ALERT_LOG" ]]; then
        echo -e "${YELLOW}🚨 Recent Alerts:${NC}"
        tail -5 "$ALERT_LOG" | while read -r line; do
            echo -e "  $line"
        done
        echo ""
    fi
    
    # System status
    echo -e "${GREEN}✅ System Status:${NC}"
    if python -c "from app.main_simple import app" 2>/dev/null; then
        echo -e "  Application: ${GREEN}HEALTHY${NC}"
    else
        echo -e "  Application: ${RED}UNHEALTHY${NC}"
    fi
    
    if docker ps | grep -q opinion-market 2>/dev/null; then
        echo -e "  Docker Container: ${GREEN}RUNNING${NC}"
    else
        echo -e "  Docker Container: ${RED}NOT RUNNING${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}💡 Press Ctrl+C to stop monitoring${NC}"
}

# Cleanup function
cleanup() {
    log "Cleaning up monitoring system..."
    # Add cleanup logic here
    echo -e "${GREEN}✅ Monitoring system stopped${NC}"
}

# Main monitoring loop
monitor_loop() {
    log "Starting advanced monitoring loop..."
    
    while true; do
        # Collect metrics
        collect_metrics
        
        # Check alerts
        if check_alerts; then
            log_warning "Alerts triggered, executing actions..."
        fi
        
        # Generate dashboard
        generate_dashboard
        
        # Wait for next check
        sleep $HEALTH_CHECK_INTERVAL
    done
}

# Help function
show_help() {
    echo "Advanced CI/CD Pipeline Monitoring System"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dashboard    Show monitoring dashboard only"
    echo "  -m, --monitor      Start continuous monitoring"
    echo "  -c, --config       Show current configuration"
    echo "  -t, --test         Test alert system"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SLACK_WEBHOOK_URL  Slack webhook for notifications"
    echo "  ALERT_EMAIL        Email address for alerts"
    echo "  PAGERDUTY_SERVICE_KEY  PagerDuty service key"
    echo ""
    echo "Examples:"
    echo "  $0 -m              # Start continuous monitoring"
    echo "  $0 -d              # Show dashboard once"
    echo "  $0 -t              # Test alert system"
}

# Test alert system
test_alerts() {
    log "Testing alert system..."
    
    # Simulate high response time
    echo "2.5" > /tmp/test_response_time
    trigger_alert "test_alert" "$WARNING" "This is a test alert"
    
    log_success "Alert system test completed"
}

# Main function
main() {
    case "${1:-}" in
        -d|--dashboard)
            init_monitoring
            collect_metrics
            generate_dashboard
            ;;
        -m|--monitor)
            init_monitoring
            monitor_loop
            ;;
        -c|--config)
            if [[ -f "$CONFIG_FILE" ]]; then
                cat "$CONFIG_FILE"
            else
                echo "No configuration file found. Run with -m to create default config."
            fi
            ;;
        -t|--test)
            init_monitoring
            test_alerts
            ;;
        -h|--help)
            show_help
            ;;
        "")
            init_monitoring
            collect_metrics
            generate_dashboard
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
