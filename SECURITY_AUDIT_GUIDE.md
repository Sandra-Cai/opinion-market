# 🔒 Security Audit Guide

## Overview

The Opinion Market platform includes a comprehensive security audit system that provides automated vulnerability scanning, compliance monitoring, and security risk assessment. This system helps maintain the highest security standards and ensures compliance with industry best practices.

## Features

### 🔍 Automated Vulnerability Scanning
- **Code Analysis**: Static analysis of source code for security vulnerabilities
- **Dependency Scanning**: Automated scanning of third-party dependencies
- **Configuration Review**: Security analysis of configuration files
- **Pattern Detection**: Advanced pattern matching for known vulnerability types

### 📊 Security Compliance Monitoring
- **OWASP Top 10**: Compliance with OWASP Top 10 security risks
- **NIST Framework**: Alignment with NIST Cybersecurity Framework
- **Industry Standards**: Support for various security standards
- **Compliance Scoring**: Automated compliance assessment and scoring

### 🚨 Risk Assessment and Management
- **Vulnerability Classification**: Categorization by severity and impact
- **Risk Scoring**: Quantitative risk assessment and prioritization
- **Trend Analysis**: Security posture tracking over time
- **Remediation Tracking**: Progress monitoring for security fixes

### 💡 Security Recommendations
- **Actionable Insights**: Specific recommendations for security improvements
- **Implementation Guidance**: Step-by-step remediation instructions
- **Best Practices**: Industry-standard security recommendations
- **Automated Fixes**: Suggestions for automated security improvements

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                 Security Audit System                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Vulnerability│  │ Compliance  │  │   Risk      │        │
│  │   Scanner    │  │  Monitor    │  │ Assessment  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Code      │  │Dependency   │  │Configuration│        │
│  │  Analysis   │  │  Scanner    │  │   Scanner   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Vulnerability Types

| Type | Severity | Description | CWE | OWASP |
|------|----------|-------------|-----|-------|
| SQL Injection | Critical | User input directly concatenated into SQL queries | CWE-89 | A03:2021 |
| Cross-Site Scripting (XSS) | High | User input rendered without sanitization | CWE-79 | A03:2021 |
| Path Traversal | High | File paths constructed using user input | CWE-22 | A01:2021 |
| Command Injection | Critical | User input passed to system commands | CWE-78 | A03:2021 |
| Weak Authentication | High | Hardcoded or easily guessable credentials | CWE-287 | A07:2021 |
| Sensitive Data Exposure | Medium | Passwords or secrets logged or exposed | CWE-200 | A02:2021 |
| Security Misconfiguration | Medium | Default or insecure settings | CWE-16 | A05:2021 |

## API Endpoints

### Security Scanning

#### Start Security Scan
```http
POST /api/v1/security/scan/start
Content-Type: application/json

{
  "target_paths": ["app/", "tests/", "scripts/"]
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Security scan started in background",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Get Scan Status
```http
GET /api/v1/security/scan/status
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "scan_in_progress": false,
    "last_scan": "2024-01-15T10:30:00Z",
    "scan_duration": 45.2
  }
}
```

### Vulnerability Management

#### Get Vulnerabilities
```http
GET /api/v1/security/vulnerabilities?severity=critical&status=open
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "vulnerabilities": [
      {
        "id": "abc123def456",
        "type": "sql_injection",
        "severity": "critical",
        "title": "SQL Injection Vulnerability",
        "description": "Potential SQL injection vulnerability detected. User input may be directly concatenated into SQL queries.",
        "file_path": "app/api/v1/endpoints/users.py",
        "line_number": 45,
        "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
        "recommendation": "Use parameterized queries or prepared statements. Never concatenate user input directly into SQL queries.",
        "cwe_id": "CWE-89",
        "owasp_category": "A03:2021 – Injection",
        "detected_at": "2024-01-15T10:30:00Z",
        "status": "open",
        "false_positive": false
      }
    ],
    "count": 1,
    "filters": {
      "severity": "critical",
      "status": "open",
      "vuln_type": null
    }
  }
}
```

#### Update Vulnerability Status
```http
PUT /api/v1/security/vulnerabilities/abc123def456/status
Content-Type: application/json

{
  "new_status": "fixed",
  "false_positive": false
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Vulnerability status updated to fixed",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Security Metrics

#### Get Security Metrics
```http
GET /api/v1/security/metrics
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_vulnerabilities": 15,
    "critical_vulnerabilities": 2,
    "high_vulnerabilities": 5,
    "medium_vulnerabilities": 6,
    "low_vulnerabilities": 2,
    "fixed_vulnerabilities": 8,
    "false_positives": 1,
    "compliance_score": 75.5,
    "last_scan": "2024-01-15T10:30:00Z",
    "scan_duration": 45.2,
    "risk_score": 2.8
  }
}
```

### Compliance Reporting

#### Get Compliance Report
```http
GET /api/v1/security/compliance?standard=OWASP_TOP_10
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "OWASP_TOP_10": {
      "total_requirements": 10,
      "compliant": 7,
      "non_compliant": 2,
      "partial": 1,
      "requirements": [
        {
          "requirement": "A01:2021 – Broken Access Control",
          "status": "compliant",
          "evidence": ["RBAC implemented", "Access control middleware active"],
          "last_checked": "2024-01-15T10:30:00Z"
        },
        {
          "requirement": "A02:2021 – Cryptographic Failures",
          "status": "partial",
          "evidence": ["TLS 1.3 implemented", "Password hashing with bcrypt"],
          "last_checked": "2024-01-15T10:30:00Z"
        }
      ]
    }
  }
}
```

### Security Dashboard

#### Get Security Dashboard
```http
GET /api/v1/security/dashboard
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "overall_security_score": 78.5,
    "risk_score": 2.8,
    "compliance_score": 75.5,
    "metrics": {
      "total_vulnerabilities": 15,
      "critical_vulnerabilities": 2,
      "high_vulnerabilities": 5,
      "medium_vulnerabilities": 6,
      "low_vulnerabilities": 2,
      "fixed_vulnerabilities": 8,
      "false_positives": 1,
      "compliance_score": 75.5,
      "last_scan": "2024-01-15T10:30:00Z",
      "scan_duration": 45.2,
      "risk_score": 2.8
    },
    "recent_vulnerabilities": [
      {
        "id": "abc123def456",
        "type": "sql_injection",
        "severity": "critical",
        "title": "SQL Injection Vulnerability",
        "file_path": "app/api/v1/endpoints/users.py",
        "line_number": 45,
        "detected_at": "2024-01-15T10:30:00Z"
      }
    ],
    "compliance_summary": {
      "OWASP_TOP_10": {
        "total_requirements": 10,
        "compliance_percentage": 70.0
      },
      "NIST_CYBERSECURITY_FRAMEWORK": {
        "total_requirements": 22,
        "compliance_percentage": 81.8
      }
    },
    "scan_status": {
      "in_progress": false,
      "last_scan": "2024-01-15T10:30:00Z"
    }
  }
}
```

## Usage Examples

### Python Integration

#### Basic Security Scanning
```python
from app.core.security_audit import security_auditor

# Run comprehensive security scan
result = await security_auditor.run_comprehensive_scan([
    "app/",
    "tests/",
    "scripts/"
])

print(f"Scan completed: {result['status']}")
print(f"Vulnerabilities found: {result['vulnerabilities_found']}")
print(f"Compliance checks: {result['compliance_checks']}")

# Get vulnerabilities by severity
critical_vulns = security_auditor.get_vulnerabilities(
    severity=SecurityLevel.CRITICAL
)

for vuln in critical_vulns:
    print(f"Critical: {vuln['title']} in {vuln['file_path']}:{vuln['line_number']}")
```

#### Vulnerability Management
```python
# Get all open vulnerabilities
open_vulns = security_auditor.get_vulnerabilities(status="open")

# Update vulnerability status
vuln_id = "abc123def456"
if vuln_id in security_auditor.vulnerabilities:
    vulnerability = security_auditor.vulnerabilities[vuln_id]
    vulnerability.status = "fixed"
    vulnerability.false_positive = False
    
    # Recalculate metrics
    security_auditor._calculate_metrics()

# Get compliance report
compliance_report = security_auditor.get_compliance_report()
for standard, data in compliance_report.items():
    print(f"{standard}: {data['compliant']}/{data['total_requirements']} compliant")
```

### API Integration

#### Security Dashboard
```javascript
// Get security dashboard
const response = await fetch('/api/v1/security/dashboard');
const dashboard = await response.json();

console.log(`Security Score: ${dashboard.data.overall_security_score}`);
console.log(`Risk Score: ${dashboard.data.risk_score}`);
console.log(`Compliance Score: ${dashboard.data.compliance_score}`);

// Display recent vulnerabilities
dashboard.data.recent_vulnerabilities.forEach(vuln => {
  console.log(`${vuln.severity.toUpperCase()}: ${vuln.title}`);
  console.log(`  File: ${vuln.file_path}:${vuln.line_number}`);
});
```

#### Vulnerability Management
```javascript
// Get critical vulnerabilities
const criticalVulns = await fetch('/api/v1/security/vulnerabilities?severity=critical');
const vulns = await criticalVulns.json();

// Update vulnerability status
for (const vuln of vulns.data.vulnerabilities) {
  await fetch(`/api/v1/security/vulnerabilities/${vuln.id}/status`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      new_status: 'fixed',
      false_positive: false
    })
  });
}
```

## Configuration

### Environment Variables

```bash
# Security Audit Configuration
SECURITY_AUDIT_ENABLED=true
SECURITY_SCAN_INTERVAL=3600
SECURITY_ALERT_THRESHOLDS='{"critical": 1, "high": 5, "medium": 10}'

# Compliance Standards
OWASP_COMPLIANCE_ENABLED=true
NIST_COMPLIANCE_ENABLED=true
CUSTOM_COMPLIANCE_STANDARDS='["ISO27001", "SOC2"]'
```

### Scan Configuration

```python
# Customize scan patterns
security_auditor.scan_patterns[VulnerabilityType.SQL_INJECTION].extend([
    r"raw_query\s*\(\s*['\"].*%.*['\"]",
    r"execute_raw\s*\(\s*['\"].*%.*['\"]"
])

# Add custom compliance standards
security_auditor.compliance_standards["CUSTOM_STANDARD"] = [
    "Requirement 1: Custom security requirement",
    "Requirement 2: Another custom requirement"
]
```

## Best Practices

### Security Scanning

1. **Regular Scanning**
   - Run security scans daily or on every code change
   - Integrate security scanning into CI/CD pipeline
   - Schedule comprehensive scans during low-traffic periods

2. **Vulnerability Management**
   - Prioritize critical and high-severity vulnerabilities
   - Track remediation progress and timelines
   - Implement false positive management

3. **Compliance Monitoring**
   - Regularly review compliance status
   - Address non-compliant requirements promptly
   - Maintain compliance documentation

### Code Security

1. **Input Validation**
   - Validate and sanitize all user input
   - Use parameterized queries for database operations
   - Implement proper output encoding

2. **Authentication and Authorization**
   - Use strong authentication mechanisms
   - Implement proper session management
   - Apply principle of least privilege

3. **Data Protection**
   - Encrypt sensitive data at rest and in transit
   - Implement proper key management
   - Use secure communication protocols

## Troubleshooting

### Common Issues

#### Scan Failures
```bash
# Check scan status
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/security/scan/status

# Review scan logs
tail -f logs/security_audit.log
```

#### High Vulnerability Count
```bash
# Get vulnerability breakdown
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/security/metrics

# Focus on critical vulnerabilities
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/security/vulnerabilities?severity=critical
```

#### Compliance Issues
```bash
# Check compliance status
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/security/compliance

# Get specific standard compliance
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/security/compliance?standard=OWASP_TOP_10
```

### Security Remediation

1. **Critical Vulnerabilities**
   - Address immediately (within 24 hours)
   - Implement temporary mitigations if needed
   - Coordinate with development team for fixes

2. **High Severity Issues**
   - Address within 1 week
   - Plan remediation in next sprint
   - Monitor for exploitation attempts

3. **Medium/Low Severity**
   - Address in regular development cycle
   - Include in technical debt backlog
   - Review during security assessments

## Integration with CI/CD

### GitHub Actions Integration

```yaml
name: Security Audit
on: [push, pull_request]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run Security Scan
        run: |
          python -m pytest tests/test_security_audit.py -v
          
      - name: Upload Security Report
        uses: actions/upload-artifact@v2
        with:
          name: security-report
          path: security_report.json
```

### Jenkins Integration

```groovy
pipeline {
    agent any
    
    stages {
        stage('Security Scan') {
            steps {
                sh 'python -m pytest tests/test_security_audit.py -v'
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'htmlcov',
                    reportFiles: 'index.html',
                    reportName: 'Security Report'
                ])
            }
        }
    }
}
```

## Security Standards

### OWASP Top 10 2021

1. **A01:2021 – Broken Access Control**
2. **A02:2021 – Cryptographic Failures**
3. **A03:2021 – Injection**
4. **A04:2021 – Insecure Design**
5. **A05:2021 – Security Misconfiguration**
6. **A06:2021 – Vulnerable and Outdated Components**
7. **A07:2021 – Identification and Authentication Failures**
8. **A08:2021 – Software and Data Integrity Failures**
9. **A09:2021 – Security Logging and Monitoring Failures**
10. **A10:2021 – Server-Side Request Forgery (SSRF)**

### NIST Cybersecurity Framework

- **Identify**: Asset Management, Business Environment, Governance
- **Protect**: Identity Management, Awareness and Training, Data Security
- **Detect**: Anomalies and Events, Security Continuous Monitoring
- **Respond**: Response Planning, Communications, Analysis, Mitigation
- **Recover**: Recovery Planning, Improvements, Communications

## Future Enhancements

- **Machine Learning**: Advanced vulnerability detection using ML
- **Real-time Monitoring**: Continuous security monitoring
- **Integration APIs**: Third-party security tool integrations
- **Automated Remediation**: Automated security fix suggestions
- **Threat Intelligence**: Integration with threat intelligence feeds

