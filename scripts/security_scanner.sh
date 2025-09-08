#!/bin/bash

# 🔒 Advanced Security Scanning & Compliance System
# Comprehensive security analysis, vulnerability scanning, and compliance checking

set -euo pipefail

# Configuration
SECURITY_LOG="/tmp/security.log"
VULNERABILITY_REPORT="/tmp/vulnerability_report.json"
COMPLIANCE_REPORT="/tmp/compliance_report.md"
SECURITY_SCORE="/tmp/security_score.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Security thresholds
CRITICAL_VULNERABILITIES=0
HIGH_VULNERABILITIES=2
MEDIUM_VULNERABILITIES=10
LOW_VULNERABILITIES=50

# Initialize security system
init_security() {
    echo -e "${PURPLE}🔒 Initializing Advanced Security Scanning System${NC}"
    
    # Install security tools
    install_security_tools
    
    # Create security reports
    echo '{"vulnerabilities": [], "compliance_checks": [], "security_score": 100}' > "$SECURITY_SCORE"
    
    echo -e "${GREEN}✅ Security system initialized${NC}"
}

# Install security scanning tools
install_security_tools() {
    log_security "Installing security scanning tools..."
    
    # Install Python security tools
    pip install --quiet bandit safety semgrep
    
    # Install system security tools
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y lynis rkhunter chkrootkit
    elif command -v brew &> /dev/null; then
        brew install lynis 2>/dev/null || true
    fi
    
    # Install Trivy for container scanning
    if ! command -v trivy &> /dev/null; then
        if command -v apt-get &> /dev/null; then
            wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
            echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
            sudo apt-get update -qq
            sudo apt-get install -y trivy
        fi
    fi
    
    log_security_success "Security tools installed"
}

# Logging functions
log_security() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$SECURITY_LOG"
}

log_security_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$SECURITY_LOG"
}

log_security_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$SECURITY_LOG"
}

log_security_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$SECURITY_LOG"
}

# Python code security scanning with Bandit
scan_python_security() {
    log_security "Scanning Python code for security vulnerabilities..."
    
    # Run Bandit security scan
    bandit -r app/ -f json -o /tmp/bandit_report.json 2>/dev/null || true
    
    # Parse Bandit results
    local critical_count=$(jq '[.results[] | select(.issue_severity == "HIGH")] | length' /tmp/bandit_report.json 2>/dev/null || echo "0")
    local medium_count=$(jq '[.results[] | select(.issue_severity == "MEDIUM")] | length' /tmp/bandit_report.json 2>/dev/null || echo "0")
    local low_count=$(jq '[.results[] | select(.issue_severity == "LOW")] | length' /tmp/bandit_report.json 2>/dev/null || echo "0")
    
    log_security "Bandit scan results: Critical=$critical_count, Medium=$medium_count, Low=$low_count"
    
    # Update security score
    jq --argjson critical "$critical_count" --argjson medium "$medium_count" --argjson low "$low_count" '
        .vulnerabilities += [{
            "tool": "bandit",
            "critical": $critical,
            "medium": $medium,
            "low": $low,
            "total": ($critical + $medium + $low)
        }]
    ' "$SECURITY_SCORE" > "${SECURITY_SCORE}.tmp" && mv "${SECURITY_SCORE}.tmp" "$SECURITY_SCORE"
    
    if [[ $critical_count -gt 0 ]]; then
        log_security_error "Critical security vulnerabilities found in Python code"
        return 1
    fi
    
    log_security_success "Python security scan completed"
    return 0
}

# Dependency vulnerability scanning with Safety
scan_dependency_vulnerabilities() {
    log_security "Scanning dependencies for known vulnerabilities..."
    
    # Run Safety scan
    safety check --json --output /tmp/safety_report.json 2>/dev/null || true
    
    # Parse Safety results
    local vulnerability_count=$(jq '.vulnerabilities | length' /tmp/safety_report.json 2>/dev/null || echo "0")
    
    log_security "Safety scan results: $vulnerability_count vulnerabilities found"
    
    # Update security score
    jq --argjson count "$vulnerability_count" '
        .vulnerabilities += [{
            "tool": "safety",
            "vulnerabilities": $count
        }]
    ' "$SECURITY_SCORE" > "${SECURITY_SCORE}.tmp" && mv "${SECURITY_SCORE}.tmp" "$SECURITY_SCORE"
    
    if [[ $vulnerability_count -gt 0 ]]; then
        log_security_warning "Dependency vulnerabilities found"
        # Display vulnerabilities
        jq -r '.vulnerabilities[] | "  - \(.package): \(.vulnerability)"' /tmp/safety_report.json 2>/dev/null | head -10
    fi
    
    log_security_success "Dependency vulnerability scan completed"
    return 0
}

# Container security scanning with Trivy
scan_container_security() {
    log_security "Scanning container images for vulnerabilities..."
    
    # Build test image if it doesn't exist
    if ! docker image inspect opinion-market:test 2>/dev/null; then
        docker build -f Dockerfile.simple -t opinion-market:test . 2>/dev/null || {
            log_security_warning "Could not build test image for container scanning"
            return 0
        }
    fi
    
    # Run Trivy scan
    if command -v trivy &> /dev/null; then
        trivy image --format json --output /tmp/trivy_report.json opinion-market:test 2>/dev/null || true
        
        # Parse Trivy results
        local critical_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' /tmp/trivy_report.json 2>/dev/null || echo "0")
        local high_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' /tmp/trivy_report.json 2>/dev/null || echo "0")
        local medium_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "MEDIUM")] | length' /tmp/trivy_report.json 2>/dev/null || echo "0")
        
        log_security "Trivy scan results: Critical=$critical_count, High=$high_count, Medium=$medium_count"
        
        # Update security score
        jq --argjson critical "$critical_count" --argjson high "$high_count" --argjson medium "$medium_count" '
            .vulnerabilities += [{
                "tool": "trivy",
                "critical": $critical,
                "high": $high,
                "medium": $medium,
                "total": ($critical + $high + $medium)
            }]
        ' "$SECURITY_SCORE" > "${SECURITY_SCORE}.tmp" && mv "${SECURITY_SCORE}.tmp" "$SECURITY_SCORE"
        
        if [[ $critical_count -gt 0 ]]; then
            log_security_error "Critical vulnerabilities found in container image"
            return 1
        fi
    else
        log_security_warning "Trivy not available, skipping container scan"
    fi
    
    log_security_success "Container security scan completed"
    return 0
}

# Configuration security analysis
analyze_configuration_security() {
    log_security "Analyzing configuration security..."
    
    local config_issues=0
    
    # Check for hardcoded secrets
    if grep -r "password\|secret\|key\|token" app/ --include="*.py" | grep -v "example\|test\|placeholder" | grep -q "="; then
        log_security_warning "Potential hardcoded secrets found in code"
        config_issues=$((config_issues + 1))
    fi
    
    # Check for debug mode in production
    if grep -r "debug.*=.*True\|DEBUG.*=.*True" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Debug mode may be enabled"
        config_issues=$((config_issues + 1))
    fi
    
    # Check for insecure HTTP
    if grep -r "http://" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Insecure HTTP URLs found"
        config_issues=$((config_issues + 1))
    fi
    
    # Check for weak encryption
    if grep -r "md5\|sha1" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Weak hashing algorithms found"
        config_issues=$((config_issues + 1))
    fi
    
    # Check for SQL injection risks
    if grep -r "execute.*%" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Potential SQL injection risks found"
        config_issues=$((config_issues + 1))
    fi
    
    # Update security score
    jq --argjson issues "$config_issues" '
        .compliance_checks += [{
            "category": "configuration",
            "issues": $issues,
            "status": (if $issues == 0 then "pass" else "fail" end)
        }]
    ' "$SECURITY_SCORE" > "${SECURITY_SCORE}.tmp" && mv "${SECURITY_SCORE}.tmp" "$SECURITY_SCORE"
    
    if [[ $config_issues -eq 0 ]]; then
        log_security_success "Configuration security analysis passed"
    else
        log_security_warning "Configuration security issues found: $config_issues"
    fi
    
    return 0
}

# Authentication and authorization analysis
analyze_auth_security() {
    log_security "Analyzing authentication and authorization..."
    
    local auth_issues=0
    
    # Check for proper authentication
    if ! grep -r "get_current_user\|authenticate" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Authentication mechanisms not found"
        auth_issues=$((auth_issues + 1))
    fi
    
    # Check for authorization checks
    if ! grep -r "has_permission\|check_permission\|authorize" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Authorization checks not found"
        auth_issues=$((auth_issues + 1))
    fi
    
    # Check for JWT token handling
    if ! grep -r "jwt\|token" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "JWT token handling not found"
        auth_issues=$((auth_issues + 1))
    fi
    
    # Check for password hashing
    if ! grep -r "bcrypt\|hash_password\|password_hash" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Password hashing not found"
        auth_issues=$((auth_issues + 1))
    fi
    
    # Update security score
    jq --argjson issues "$auth_issues" '
        .compliance_checks += [{
            "category": "authentication",
            "issues": $issues,
            "status": (if $issues == 0 then "pass" else "fail" end)
        }]
    ' "$SECURITY_SCORE" > "${SECURITY_SCORE}.tmp" && mv "${SECURITY_SCORE}.tmp" "$SECURITY_SCORE"
    
    if [[ $auth_issues -eq 0 ]]; then
        log_security_success "Authentication security analysis passed"
    else
        log_security_warning "Authentication security issues found: $auth_issues"
    fi
    
    return 0
}

# Data protection analysis
analyze_data_protection() {
    log_security "Analyzing data protection measures..."
    
    local data_issues=0
    
    # Check for data encryption
    if ! grep -r "encrypt\|cipher" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Data encryption not found"
        data_issues=$((data_issues + 1))
    fi
    
    # Check for input validation
    if ! grep -r "validate\|sanitize" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Input validation not found"
        data_issues=$((data_issues + 1))
    fi
    
    # Check for data sanitization
    if ! grep -r "escape\|clean" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Data sanitization not found"
        data_issues=$((data_issues + 1))
    fi
    
    # Check for secure headers
    if ! grep -r "X-Frame-Options\|X-Content-Type-Options\|X-XSS-Protection" app/ --include="*.py" 2>/dev/null; then
        log_security_warning "Security headers not found"
        data_issues=$((data_issues + 1))
    fi
    
    # Update security score
    jq --argjson issues "$data_issues" '
        .compliance_checks += [{
            "category": "data_protection",
            "issues": $issues,
            "status": (if $issues == 0 then "pass" else "fail" end)
        }]
    ' "$SECURITY_SCORE" > "${SECURITY_SCORE}.tmp" && mv "${SECURITY_SCORE}.tmp" "$SECURITY_SCORE"
    
    if [[ $data_issues -eq 0 ]]; then
        log_security_success "Data protection analysis passed"
    else
        log_security_warning "Data protection issues found: $data_issues"
    fi
    
    return 0
}

# Calculate security score
calculate_security_score() {
    log_security "Calculating overall security score..."
    
    local total_issues=0
    local total_checks=0
    
    # Count vulnerabilities
    local bandit_issues=$(jq '[.vulnerabilities[] | select(.tool=="bandit") | .critical + .medium + .low] | add' "$SECURITY_SCORE" 2>/dev/null || echo "0")
    local safety_issues=$(jq '[.vulnerabilities[] | select(.tool=="safety") | .vulnerabilities] | add' "$SECURITY_SCORE" 2>/dev/null || echo "0")
    local trivy_issues=$(jq '[.vulnerabilities[] | select(.tool=="trivy") | .critical + .high + .medium] | add' "$SECURITY_SCORE" 2>/dev/null || echo "0")
    
    # Count compliance issues
    local config_issues=$(jq '[.compliance_checks[] | select(.category=="configuration") | .issues] | add' "$SECURITY_SCORE" 2>/dev/null || echo "0")
    local auth_issues=$(jq '[.compliance_checks[] | select(.category=="authentication") | .issues] | add' "$SECURITY_SCORE" 2>/dev/null || echo "0")
    local data_issues=$(jq '[.compliance_checks[] | select(.category=="data_protection") | .issues] | add' "$SECURITY_SCORE" 2>/dev/null || echo "0")
    
    total_issues=$((bandit_issues + safety_issues + trivy_issues + config_issues + auth_issues + data_issues))
    total_checks=6  # Number of security checks performed
    
    # Calculate score (100 - (issues * 5))
    local score=$((100 - (total_issues * 5)))
    if [[ $score -lt 0 ]]; then
        score=0
    fi
    
    # Update security score
    jq --argjson score "$score" --argjson issues "$total_issues" '
        .security_score = $score |
        .total_issues = $issues
    ' "$SECURITY_SCORE" > "${SECURITY_SCORE}.tmp" && mv "${SECURITY_SCORE}.tmp" "$SECURITY_SCORE"
    
    log_security "Security score calculated: $score/100 (Issues: $total_issues)"
    
    echo $score
}

# Generate security report
generate_security_report() {
    log_security "Generating security report..."
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local security_score=$(calculate_security_score)
    
    cat > "$COMPLIANCE_REPORT" << EOF
# 🔒 Security & Compliance Report

**Generated:** $timestamp  
**Overall Security Score:** $security_score/100  
**System:** $(uname -s) $(uname -r)  
**Python:** $(python --version)  

## 📊 Vulnerability Summary

EOF
    
    # Add vulnerability details
    jq -r '.vulnerabilities[] | "### \(.tool | ascii_upcase) Scan

- **Critical:** \(.critical // 0)
- **High:** \(.high // 0)  
- **Medium:** \(.medium // 0)
- **Low:** \(.low // 0)
- **Total:** \(.total // .vulnerabilities // 0)

"' "$SECURITY_SCORE" >> "$COMPLIANCE_REPORT"
    
    # Add compliance check results
    echo "## ✅ Compliance Checks" >> "$COMPLIANCE_REPORT"
    echo "" >> "$COMPLIANCE_REPORT"
    
    jq -r '.compliance_checks[] | "- **\(.category | ascii_upcase | gsub("_"; " ")):** \(.status | ascii_upcase) (\(.issues) issues)"' "$SECURITY_SCORE" >> "$COMPLIANCE_REPORT"
    
    # Add security recommendations
    cat >> "$COMPLIANCE_REPORT" << EOF

## 🎯 Security Recommendations

1. **Regular Updates:** Keep all dependencies updated
2. **Code Review:** Implement mandatory security code reviews
3. **Automated Scanning:** Integrate security scanning into CI/CD
4. **Access Control:** Implement proper authentication and authorization
5. **Data Protection:** Encrypt sensitive data at rest and in transit
6. **Monitoring:** Set up security monitoring and alerting
7. **Incident Response:** Establish security incident response procedures

## 📈 Security Trends

*Historical security data would be displayed here in a production system.*

## 🔧 Next Steps

1. Address critical and high severity vulnerabilities immediately
2. Implement recommended security measures
3. Set up continuous security monitoring
4. Conduct regular security assessments
5. Train development team on security best practices

EOF
    
    log_security_success "Security report generated: $COMPLIANCE_REPORT"
}

# Run comprehensive security scan
run_security_scan() {
    log_security "Starting comprehensive security scan..."
    
    local failed_scans=0
    
    # Run all security scans
    if ! scan_python_security; then
        failed_scans=$((failed_scans + 1))
    fi
    
    if ! scan_dependency_vulnerabilities; then
        failed_scans=$((failed_scans + 1))
    fi
    
    if ! scan_container_security; then
        failed_scans=$((failed_scans + 1))
    fi
    
    if ! analyze_configuration_security; then
        failed_scans=$((failed_scans + 1))
    fi
    
    if ! analyze_auth_security; then
        failed_scans=$((failed_scans + 1))
    fi
    
    if ! analyze_data_protection; then
        failed_scans=$((failed_scans + 1))
    fi
    
    # Calculate and display security score
    local security_score=$(calculate_security_score)
    
    # Generate report
    generate_security_report
    
    # Summary
    echo ""
    echo -e "${PURPLE}🔒 Security Scan Summary${NC}"
    echo -e "Failed scans: $failed_scans"
    echo -e "Security score: $security_score/100"
    
    if [[ $security_score -ge 90 ]]; then
        log_security_success "Excellent security posture!"
        echo -e "${GREEN}🎉 Security is excellent!${NC}"
    elif [[ $security_score -ge 70 ]]; then
        log_security_warning "Good security posture with room for improvement"
        echo -e "${YELLOW}⚠️ Security is good but can be improved${NC}"
    else
        log_security_error "Security issues need immediate attention"
        echo -e "${RED}❌ Security needs immediate attention${NC}"
    fi
    
    echo -e "${CYAN}📄 Detailed report: $COMPLIANCE_REPORT${NC}"
    echo -e "${CYAN}📊 Security data: $SECURITY_SCORE${NC}"
}

# Help function
show_help() {
    echo "Advanced Security Scanning & Compliance System"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  scan        Run comprehensive security scan"
    echo "  python      Scan Python code for vulnerabilities"
    echo "  dependencies Scan dependencies for vulnerabilities"
    echo "  container   Scan container images for vulnerabilities"
    echo "  config      Analyze configuration security"
    echo "  auth        Analyze authentication security"
    echo "  data        Analyze data protection"
    echo "  report      Generate security report"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 scan     # Run full security scan"
    echo "  $0 python   # Scan Python code only"
    echo "  $0 report   # Generate security report"
}

# Main function
main() {
    case "${1:-}" in
        scan)
            init_security
            run_security_scan
            ;;
        python)
            init_security
            scan_python_security
            ;;
        dependencies)
            init_security
            scan_dependency_vulnerabilities
            ;;
        container)
            init_security
            scan_container_security
            ;;
        config)
            init_security
            analyze_configuration_security
            ;;
        auth)
            init_security
            analyze_auth_security
            ;;
        data)
            init_security
            analyze_data_protection
            ;;
        report)
            generate_security_report
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            init_security
            run_security_scan
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
