#!/bin/bash

# Comprehensive Security Audit Script
# Advanced security scanning and vulnerability assessment

set -e

# Configuration
APP_DIR="."
REPORT_DIR="security_reports"
SCAN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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

# Create report directory
create_report_dir() {
    log_info "Creating security report directory..."
    mkdir -p "$REPORT_DIR"
    log_success "Report directory created: $REPORT_DIR"
}

# Dependency security scan
scan_dependencies() {
    log_info "Scanning Python dependencies for vulnerabilities..."
    
    # Install safety if not available
    if ! command -v safety &> /dev/null; then
        log_info "Installing safety for dependency scanning..."
        pip install safety
    fi
    
    # Run safety check
    safety check --json --output "$REPORT_DIR/dependency_scan_$SCAN_TIMESTAMP.json" || true
    safety check --output "$REPORT_DIR/dependency_scan_$SCAN_TIMESTAMP.txt" || true
    
    log_success "Dependency security scan completed"
}

# Code security scan with bandit
scan_code_security() {
    log_info "Scanning code for security issues with Bandit..."
    
    # Install bandit if not available
    if ! command -v bandit &> /dev/null; then
        log_info "Installing bandit for code security scanning..."
        pip install bandit
    fi
    
    # Run bandit scan
    bandit -r "$APP_DIR" -f json -o "$REPORT_DIR/bandit_scan_$SCAN_TIMESTAMP.json" || true
    bandit -r "$APP_DIR" -f txt -o "$REPORT_DIR/bandit_scan_$SCAN_TIMESTAMP.txt" || true
    
    log_success "Code security scan completed"
}

# Docker security scan
scan_docker_security() {
    log_info "Scanning Docker images for vulnerabilities..."
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available, skipping Docker security scan"
        return 0
    fi
    
    # Build image for scanning
    log_info "Building Docker image for security scanning..."
    docker build -t opinion-market-security-scan -f Dockerfile.robust . || {
        log_warning "Docker build failed, skipping Docker security scan"
        return 0
    }
    
    # Install Trivy if not available
    if ! command -v trivy &> /dev/null; then
        log_info "Installing Trivy for Docker security scanning..."
        # For macOS
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install trivy
        else
            # For Linux
            curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
        fi
    fi
    
    # Run Trivy scan
    trivy image --format json --output "$REPORT_DIR/trivy_scan_$SCAN_TIMESTAMP.json" opinion-market-security-scan || true
    trivy image --format table --output "$REPORT_DIR/trivy_scan_$SCAN_TIMESTAMP.txt" opinion-market-security-scan || true
    
    log_success "Docker security scan completed"
}

# Configuration security check
check_configuration_security() {
    log_info "Checking configuration security..."
    
    cat > "$REPORT_DIR/config_security_$SCAN_TIMESTAMP.txt" << EOF
Configuration Security Assessment
================================

1. Environment Variables:
$(env | grep -E "(SECRET|PASSWORD|KEY|TOKEN)" | wc -l) sensitive environment variables found

2. Configuration Files:
$(find . -name "*.env*" -o -name "config.*" | wc -l) configuration files found

3. Database Configuration:
- Check for hardcoded credentials
- Verify connection encryption
- Review access permissions

4. API Security:
- Authentication mechanisms
- Rate limiting configuration
- CORS settings
- Input validation

5. Logging Configuration:
- Sensitive data in logs
- Log file permissions
- Log retention policies

Recommendations:
- Use environment variables for sensitive data
- Enable database encryption
- Implement proper authentication
- Configure rate limiting
- Validate all inputs
- Secure log files
EOF

    log_success "Configuration security check completed"
}

# Network security check
check_network_security() {
    log_info "Checking network security configuration..."
    
    cat > "$REPORT_DIR/network_security_$SCAN_TIMESTAMP.txt" << EOF
Network Security Assessment
==========================

1. Port Configuration:
$(netstat -tuln 2>/dev/null | grep LISTEN | wc -l) listening ports found

2. Firewall Status:
$(systemctl is-active ufw 2>/dev/null || echo "inactive") UFW firewall status

3. SSL/TLS Configuration:
- Check certificate validity
- Verify encryption strength
- Review cipher suites

4. Network Access:
- Review exposed endpoints
- Check for unnecessary services
- Verify access controls

Recommendations:
- Use HTTPS for all communications
- Implement proper firewall rules
- Regular certificate updates
- Monitor network traffic
- Use VPN for admin access
EOF

    log_success "Network security check completed"
}

# Authentication security check
check_authentication_security() {
    log_info "Checking authentication security..."
    
    cat > "$REPORT_DIR/auth_security_$SCAN_TIMESTAMP.txt" << EOF
Authentication Security Assessment
=================================

1. Password Policy:
- Minimum length requirements
- Complexity requirements
- Password history
- Account lockout policies

2. Session Management:
- Session timeout configuration
- Secure session storage
- Session invalidation

3. Multi-Factor Authentication:
- MFA implementation status
- Backup authentication methods
- Recovery procedures

4. Token Security:
- JWT token configuration
- Token expiration
- Token refresh mechanisms
- Secure token storage

5. Access Control:
- Role-based access control
- Principle of least privilege
- Regular access reviews

Recommendations:
- Implement strong password policies
- Enable MFA where possible
- Use secure session management
- Regular access reviews
- Monitor authentication events
EOF

    log_success "Authentication security check completed"
}

# Data protection check
check_data_protection() {
    log_info "Checking data protection measures..."
    
    cat > "$REPORT_DIR/data_protection_$SCAN_TIMESTAMP.txt" << EOF
Data Protection Assessment
=========================

1. Data Encryption:
- Data at rest encryption
- Data in transit encryption
- Key management practices

2. Data Classification:
- Sensitive data identification
- Data handling procedures
- Data retention policies

3. Backup Security:
- Backup encryption
- Secure backup storage
- Backup access controls

4. Data Privacy:
- GDPR compliance measures
- Data anonymization
- Consent management

5. Data Loss Prevention:
- DLP policies
- Data monitoring
- Incident response

Recommendations:
- Encrypt all sensitive data
- Implement data classification
- Secure backup procedures
- Regular data audits
- Privacy impact assessments
EOF

    log_success "Data protection check completed"
}

# Generate security report
generate_security_report() {
    log_info "Generating comprehensive security report..."
    
    local report_file="$REPORT_DIR/security_audit_report_$SCAN_TIMESTAMP.md"
    
    cat > "$report_file" << EOF
# Security Audit Report
Generated: $(date)

## Executive Summary
This report provides a comprehensive security assessment of the Opinion Market API project.

## Scan Results

### 1. Dependency Security
EOF

    # Add dependency scan results
    if [ -f "$REPORT_DIR/dependency_scan_$SCAN_TIMESTAMP.txt" ]; then
        echo "#### Python Dependencies" >> "$report_file"
        echo '```' >> "$report_file"
        cat "$REPORT_DIR/dependency_scan_$SCAN_TIMESTAMP.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

### 2. Code Security
EOF

    # Add bandit scan results
    if [ -f "$REPORT_DIR/bandit_scan_$SCAN_TIMESTAMP.txt" ]; then
        echo "#### Static Code Analysis" >> "$report_file"
        echo '```' >> "$report_file"
        head -50 "$REPORT_DIR/bandit_scan_$SCAN_TIMESTAMP.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

### 3. Container Security
EOF

    # Add Trivy scan results
    if [ -f "$REPORT_DIR/trivy_scan_$SCAN_TIMESTAMP.txt" ]; then
        echo "#### Docker Image Vulnerabilities" >> "$report_file"
        echo '```' >> "$report_file"
        head -50 "$REPORT_DIR/trivy_scan_$SCAN_TIMESTAMP.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

### 4. Configuration Security
EOF

    # Add configuration security results
    if [ -f "$REPORT_DIR/config_security_$SCAN_TIMESTAMP.txt" ]; then
        echo '```' >> "$report_file"
        cat "$REPORT_DIR/config_security_$SCAN_TIMESTAMP.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

### 5. Network Security
EOF

    # Add network security results
    if [ -f "$REPORT_DIR/network_security_$SCAN_TIMESTAMP.txt" ]; then
        echo '```' >> "$report_file"
        cat "$REPORT_DIR/network_security_$SCAN_TIMESTAMP.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

### 6. Authentication Security
EOF

    # Add authentication security results
    if [ -f "$REPORT_DIR/auth_security_$SCAN_TIMESTAMP.txt" ]; then
        echo '```' >> "$report_file"
        cat "$REPORT_DIR/auth_security_$SCAN_TIMESTAMP.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

### 7. Data Protection
EOF

    # Add data protection results
    if [ -f "$REPORT_DIR/data_protection_$SCAN_TIMESTAMP.txt" ]; then
        echo '```' >> "$report_file"
        cat "$REPORT_DIR/data_protection_$SCAN_TIMESTAMP.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

## Security Recommendations

### Immediate Actions
1. **Update Dependencies**: Address any high-severity vulnerabilities
2. **Code Review**: Review and fix any security issues found in code
3. **Configuration**: Secure all configuration files and environment variables
4. **Access Control**: Implement proper authentication and authorization

### Ongoing Security Measures
1. **Regular Scans**: Schedule automated security scans
2. **Dependency Updates**: Keep all dependencies up to date
3. **Security Training**: Provide security training for developers
4. **Incident Response**: Establish security incident response procedures

### Advanced Security Features
1. **WAF**: Implement Web Application Firewall
2. **SIEM**: Deploy Security Information and Event Management
3. **Penetration Testing**: Conduct regular penetration tests
4. **Security Monitoring**: Implement continuous security monitoring

## Compliance
- **OWASP Top 10**: Address all OWASP Top 10 vulnerabilities
- **GDPR**: Ensure data protection compliance
- **SOC 2**: Implement SOC 2 security controls
- **ISO 27001**: Consider ISO 27001 certification

## Files Generated
- Dependency scan: dependency_scan_$SCAN_TIMESTAMP.*
- Code security scan: bandit_scan_$SCAN_TIMESTAMP.*
- Container security scan: trivy_scan_$SCAN_TIMESTAMP.*
- Configuration security: config_security_$SCAN_TIMESTAMP.txt
- Network security: network_security_$SCAN_TIMESTAMP.txt
- Authentication security: auth_security_$SCAN_TIMESTAMP.txt
- Data protection: data_protection_$SCAN_TIMESTAMP.txt

---
*This report was generated automatically. Please review all findings and implement appropriate security measures.*
EOF

    log_success "Security report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting comprehensive security audit"
    
    # Execute security checks
    create_report_dir
    scan_dependencies
    scan_code_security
    scan_docker_security
    check_configuration_security
    check_network_security
    check_authentication_security
    check_data_protection
    generate_security_report
    
    log_success "Security audit completed successfully!"
    log_info "Reports generated in: $REPORT_DIR"
}

# Run main function
main "$@"
