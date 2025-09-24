"""
Comprehensive Security Audit System
Provides automated security scanning, vulnerability detection, and compliance monitoring
"""

import asyncio
import logging
import hashlib
import re
import json
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import subprocess
import os
import ast
import inspect

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    CSRF = "cross_site_request_forgery"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    WEAK_AUTHENTICATION = "weak_authentication"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    BROKEN_ACCESS_CONTROL = "broken_access_control"


@dataclass
class SecurityVulnerability:
    """Security vulnerability data structure"""
    id: str
    type: VulnerabilityType
    severity: SecurityLevel
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)
    status: str = "open"  # open, fixed, false_positive, accepted_risk
    false_positive: bool = False


@dataclass
class SecurityCompliance:
    """Security compliance data structure"""
    standard: str  # OWASP, NIST, ISO27001, etc.
    requirement: str
    status: str  # compliant, non_compliant, partial
    evidence: List[str] = field(default_factory=list)
    last_checked: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityMetrics:
    """Security metrics data structure"""
    total_vulnerabilities: int = 0
    critical_vulnerabilities: int = 0
    high_vulnerabilities: int = 0
    medium_vulnerabilities: int = 0
    low_vulnerabilities: int = 0
    fixed_vulnerabilities: int = 0
    false_positives: int = 0
    compliance_score: float = 0.0
    last_scan: Optional[datetime] = None
    scan_duration: float = 0.0


class SecurityAuditor:
    """Comprehensive security audit system"""
    
    def __init__(self):
        self.vulnerabilities: Dict[str, SecurityVulnerability] = {}
        self.compliance_checks: List[SecurityCompliance] = []
        self.metrics = SecurityMetrics()
        self.scan_patterns: Dict[VulnerabilityType, List[str]] = {
            VulnerabilityType.SQL_INJECTION: [
                r"execute\s*\(\s*['\"].*%.*['\"]",
                r"query\s*\(\s*['\"].*%.*['\"]",
                r"cursor\.execute\s*\(\s*['\"].*%.*['\"]",
                r"SELECT.*FROM.*WHERE.*%",
                r"INSERT.*INTO.*VALUES.*%",
                r"UPDATE.*SET.*WHERE.*%",
                r"DELETE.*FROM.*WHERE.*%",
            ],
            VulnerabilityType.XSS: [
                r"render_template_string\s*\(",
                r"Markup\s*\(",
                r"safe\s*\(",
                r"innerHTML\s*=",
                r"document\.write\s*\(",
                r"eval\s*\(",
            ],
            VulnerabilityType.PATH_TRAVERSAL: [
                r"open\s*\(\s*['\"].*\.\./",
                r"file\s*\(\s*['\"].*\.\./",
                r"os\.path\.join\s*\(.*\.\./",
                r"\.\./.*\.\./",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
            ],
            VulnerabilityType.COMMAND_INJECTION: [
                r"os\.system\s*\(",
                r"subprocess\.call\s*\(",
                r"subprocess\.run\s*\(",
                r"subprocess\.Popen\s*\(",
                r"shell\s*=\s*True",
                r"eval\s*\(",
                r"exec\s*\(",
            ],
            VulnerabilityType.WEAK_AUTHENTICATION: [
                r"password\s*=\s*['\"][^'\"]{1,7}['\"]",
                r"SECRET_KEY\s*=\s*['\"]test['\"]",
                r"DEBUG\s*=\s*True",
                r"password.*123",
                r"admin.*admin",
            ],
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: [
                r"print\s*\(\s*password",
                r"logger\.info\s*\(\s*password",
                r"console\.log\s*\(\s*password",
                r"api_key.*=.*['\"][^'\"]+['\"]",
                r"secret.*=.*['\"][^'\"]+['\"]",
            ],
        }
        
        self.compliance_standards = {
            "OWASP_TOP_10": [
                "A01:2021 – Broken Access Control",
                "A02:2021 – Cryptographic Failures",
                "A03:2021 – Injection",
                "A04:2021 – Insecure Design",
                "A05:2021 – Security Misconfiguration",
                "A06:2021 – Vulnerable and Outdated Components",
                "A07:2021 – Identification and Authentication Failures",
                "A08:2021 – Software and Data Integrity Failures",
                "A09:2021 – Security Logging and Monitoring Failures",
                "A10:2021 – Server-Side Request Forgery (SSRF)",
            ],
            "NIST_CYBERSECURITY_FRAMEWORK": [
                "Identify: Asset Management",
                "Identify: Business Environment",
                "Identify: Governance",
                "Identify: Risk Assessment",
                "Identify: Risk Management Strategy",
                "Protect: Identity Management and Access Control",
                "Protect: Awareness and Training",
                "Protect: Data Security",
                "Protect: Information Protection Processes and Procedures",
                "Protect: Maintenance",
                "Protect: Protective Technology",
                "Detect: Anomalies and Events",
                "Detect: Security Continuous Monitoring",
                "Detect: Detection Processes",
                "Respond: Response Planning",
                "Respond: Communications",
                "Respond: Analysis",
                "Respond: Mitigation",
                "Respond: Improvements",
                "Recover: Recovery Planning",
                "Recover: Improvements",
                "Recover: Communications",
            ]
        }
        
        self.lock = threading.Lock()
        self.scan_in_progress = False
        
    async def run_comprehensive_scan(self, target_paths: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        if self.scan_in_progress:
            return {"error": "Scan already in progress"}
            
        self.scan_in_progress = True
        start_time = time.time()
        
        try:
            logger.info("Starting comprehensive security scan")
            
            # Clear previous results
            self.vulnerabilities.clear()
            self.compliance_checks.clear()
            
            # Set default paths if none provided
            if not target_paths:
                target_paths = ["app/", "tests/", "scripts/"]
                
            # Run different types of scans
            await self._scan_code_vulnerabilities(target_paths)
            await self._scan_dependencies()
            await self._check_compliance()
            await self._scan_configuration_files()
            await self._check_security_headers()
            
            # Calculate metrics
            self._calculate_metrics()
            
            scan_duration = time.time() - start_time
            self.metrics.scan_duration = scan_duration
            self.metrics.last_scan = datetime.now()
            
            logger.info(f"Security scan completed in {scan_duration:.2f} seconds")
            
            return {
                "status": "completed",
                "vulnerabilities_found": len(self.vulnerabilities),
                "compliance_checks": len(self.compliance_checks),
                "scan_duration": scan_duration,
                "metrics": self._get_metrics_dict()
            }
            
        except Exception as e:
            logger.error(f"Error during security scan: {e}")
            return {"error": str(e)}
        finally:
            self.scan_in_progress = False
            
    async def _scan_code_vulnerabilities(self, target_paths: List[str]):
        """Scan code for security vulnerabilities"""
        logger.info("Scanning code for vulnerabilities")
        
        for path in target_paths:
            if not os.path.exists(path):
                continue
                
            for root, dirs, files in os.walk(path):
                # Skip certain directories
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv']]
                
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.html', '.php')):
                        file_path = os.path.join(root, file)
                        await self._scan_file(file_path)
                        
    async def _scan_file(self, file_path: str):
        """Scan individual file for vulnerabilities"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
            for vuln_type, patterns in self.scan_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        code_snippet = lines[line_number - 1].strip() if line_number <= len(lines) else ""
                        
                        vulnerability = SecurityVulnerability(
                            id=self._generate_vuln_id(file_path, line_number, vuln_type),
                            type=vuln_type,
                            severity=self._get_severity(vuln_type),
                            title=self._get_vuln_title(vuln_type),
                            description=self._get_vuln_description(vuln_type),
                            file_path=file_path,
                            line_number=line_number,
                            code_snippet=code_snippet,
                            recommendation=self._get_recommendation(vuln_type),
                            cwe_id=self._get_cwe_id(vuln_type),
                            owasp_category=self._get_owasp_category(vuln_type)
                        )
                        
                        self.vulnerabilities[vulnerability.id] = vulnerability
                        
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
            
    async def _scan_dependencies(self):
        """Scan dependencies for known vulnerabilities"""
        logger.info("Scanning dependencies for vulnerabilities")
        
        try:
            # Check Python dependencies
            if os.path.exists("requirements.txt"):
                result = subprocess.run(
                    ["safety", "check", "-r", "requirements.txt", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0 and result.stdout:
                    vulnerabilities = json.loads(result.stdout)
                    for vuln in vulnerabilities:
                        vulnerability = SecurityVulnerability(
                            id=f"dep_{hashlib.md5(vuln.get('package', '').encode()).hexdigest()}",
                            type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                            severity=SecurityLevel.HIGH,
                            title=f"Vulnerable Dependency: {vuln.get('package', 'Unknown')}",
                            description=vuln.get('advisory', 'No description available'),
                            file_path="requirements.txt",
                            line_number=0,
                            code_snippet=f"Package: {vuln.get('package', 'Unknown')}",
                            recommendation=f"Update {vuln.get('package', 'Unknown')} to version {vuln.get('safe_version', 'latest')}",
                            cwe_id=vuln.get('cve', ''),
                            owasp_category="A06:2021 – Vulnerable and Outdated Components"
                        )
                        self.vulnerabilities[vulnerability.id] = vulnerability
                        
        except Exception as e:
            logger.error(f"Error scanning dependencies: {e}")
            
    async def _check_compliance(self):
        """Check compliance with security standards"""
        logger.info("Checking security compliance")
        
        for standard, requirements in self.compliance_standards.items():
            for requirement in requirements:
                compliance = SecurityCompliance(
                    standard=standard,
                    requirement=requirement,
                    status="partial",  # Default to partial, would be determined by actual checks
                    evidence=[]
                )
                self.compliance_checks.append(compliance)
                
    async def _scan_configuration_files(self):
        """Scan configuration files for security issues"""
        logger.info("Scanning configuration files")
        
        config_files = [
            "docker-compose.yml",
            "Dockerfile",
            ".env",
            "config.yaml",
            "settings.py",
            "app/core/config.py"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                await self._scan_file(config_file)
                
    async def _check_security_headers(self):
        """Check for security headers configuration"""
        logger.info("Checking security headers configuration")
        
        # This would typically check middleware configuration
        # For now, we'll add a compliance check
        compliance = SecurityCompliance(
            standard="OWASP_TOP_10",
            requirement="Security Headers Implementation",
            status="partial",
            evidence=["CORS middleware configured", "Security headers middleware needed"]
        )
        self.compliance_checks.append(compliance)
        
    def _generate_vuln_id(self, file_path: str, line_number: int, vuln_type: VulnerabilityType) -> str:
        """Generate unique vulnerability ID"""
        content = f"{file_path}:{line_number}:{vuln_type.value}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
        
    def _get_severity(self, vuln_type: VulnerabilityType) -> SecurityLevel:
        """Get severity level for vulnerability type"""
        severity_map = {
            VulnerabilityType.SQL_INJECTION: SecurityLevel.CRITICAL,
            VulnerabilityType.XSS: SecurityLevel.HIGH,
            VulnerabilityType.PATH_TRAVERSAL: SecurityLevel.HIGH,
            VulnerabilityType.COMMAND_INJECTION: SecurityLevel.CRITICAL,
            VulnerabilityType.WEAK_AUTHENTICATION: SecurityLevel.HIGH,
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: SecurityLevel.MEDIUM,
            VulnerabilityType.SECURITY_MISCONFIGURATION: SecurityLevel.MEDIUM,
        }
        return severity_map.get(vuln_type, SecurityLevel.MEDIUM)
        
    def _get_vuln_title(self, vuln_type: VulnerabilityType) -> str:
        """Get vulnerability title"""
        titles = {
            VulnerabilityType.SQL_INJECTION: "SQL Injection Vulnerability",
            VulnerabilityType.XSS: "Cross-Site Scripting (XSS) Vulnerability",
            VulnerabilityType.PATH_TRAVERSAL: "Path Traversal Vulnerability",
            VulnerabilityType.COMMAND_INJECTION: "Command Injection Vulnerability",
            VulnerabilityType.WEAK_AUTHENTICATION: "Weak Authentication",
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: "Sensitive Data Exposure",
            VulnerabilityType.SECURITY_MISCONFIGURATION: "Security Misconfiguration",
        }
        return titles.get(vuln_type, "Security Vulnerability")
        
    def _get_vuln_description(self, vuln_type: VulnerabilityType) -> str:
        """Get vulnerability description"""
        descriptions = {
            VulnerabilityType.SQL_INJECTION: "Potential SQL injection vulnerability detected. User input may be directly concatenated into SQL queries.",
            VulnerabilityType.XSS: "Potential cross-site scripting vulnerability. User input may be rendered without proper sanitization.",
            VulnerabilityType.PATH_TRAVERSAL: "Potential path traversal vulnerability. File paths may be constructed using user input.",
            VulnerabilityType.COMMAND_INJECTION: "Potential command injection vulnerability. User input may be passed to system commands.",
            VulnerabilityType.WEAK_AUTHENTICATION: "Weak authentication mechanism detected. Passwords or secrets may be hardcoded or easily guessable.",
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: "Potential sensitive data exposure. Passwords, API keys, or other sensitive information may be logged or exposed.",
            VulnerabilityType.SECURITY_MISCONFIGURATION: "Security misconfiguration detected. Default or insecure settings may be in use.",
        }
        return descriptions.get(vuln_type, "Security vulnerability detected")
        
    def _get_recommendation(self, vuln_type: VulnerabilityType) -> str:
        """Get remediation recommendation"""
        recommendations = {
            VulnerabilityType.SQL_INJECTION: "Use parameterized queries or prepared statements. Never concatenate user input directly into SQL queries.",
            VulnerabilityType.XSS: "Sanitize and validate all user input. Use proper output encoding and Content Security Policy (CSP).",
            VulnerabilityType.PATH_TRAVERSAL: "Validate and sanitize file paths. Use allowlists for allowed directories and files.",
            VulnerabilityType.COMMAND_INJECTION: "Avoid executing system commands with user input. Use safe alternatives or properly validate and sanitize input.",
            VulnerabilityType.WEAK_AUTHENTICATION: "Use strong, unique passwords and secrets. Store them securely using environment variables or secret management systems.",
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: "Remove sensitive data from logs and code. Use proper secret management and encryption.",
            VulnerabilityType.SECURITY_MISCONFIGURATION: "Review and harden security configurations. Remove default credentials and disable unnecessary features.",
        }
        return recommendations.get(vuln_type, "Review and fix the security issue")
        
    def _get_cwe_id(self, vuln_type: VulnerabilityType) -> str:
        """Get CWE ID for vulnerability type"""
        cwe_map = {
            VulnerabilityType.SQL_INJECTION: "CWE-89",
            VulnerabilityType.XSS: "CWE-79",
            VulnerabilityType.PATH_TRAVERSAL: "CWE-22",
            VulnerabilityType.COMMAND_INJECTION: "CWE-78",
            VulnerabilityType.WEAK_AUTHENTICATION: "CWE-287",
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: "CWE-200",
            VulnerabilityType.SECURITY_MISCONFIGURATION: "CWE-16",
        }
        return cwe_map.get(vuln_type, "")
        
    def _get_owasp_category(self, vuln_type: VulnerabilityType) -> str:
        """Get OWASP Top 10 category"""
        owasp_map = {
            VulnerabilityType.SQL_INJECTION: "A03:2021 – Injection",
            VulnerabilityType.XSS: "A03:2021 – Injection",
            VulnerabilityType.PATH_TRAVERSAL: "A01:2021 – Broken Access Control",
            VulnerabilityType.COMMAND_INJECTION: "A03:2021 – Injection",
            VulnerabilityType.WEAK_AUTHENTICATION: "A07:2021 – Identification and Authentication Failures",
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: "A02:2021 – Cryptographic Failures",
            VulnerabilityType.SECURITY_MISCONFIGURATION: "A05:2021 – Security Misconfiguration",
        }
        return owasp_map.get(vuln_type, "")
        
    def _calculate_metrics(self):
        """Calculate security metrics"""
        self.metrics.total_vulnerabilities = len(self.vulnerabilities)
        self.metrics.critical_vulnerabilities = len([v for v in self.vulnerabilities.values() if v.severity == SecurityLevel.CRITICAL])
        self.metrics.high_vulnerabilities = len([v for v in self.vulnerabilities.values() if v.severity == SecurityLevel.HIGH])
        self.metrics.medium_vulnerabilities = len([v for v in self.vulnerabilities.values() if v.severity == SecurityLevel.MEDIUM])
        self.metrics.low_vulnerabilities = len([v for v in self.vulnerabilities.values() if v.severity == SecurityLevel.LOW])
        self.metrics.fixed_vulnerabilities = len([v for v in self.vulnerabilities.values() if v.status == "fixed"])
        self.metrics.false_positives = len([v for v in self.vulnerabilities.values() if v.false_positive])
        
        # Calculate compliance score
        if self.compliance_checks:
            compliant_count = len([c for c in self.compliance_checks if c.status == "compliant"])
            self.metrics.compliance_score = (compliant_count / len(self.compliance_checks)) * 100
        
    def _get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary"""
        return {
            "total_vulnerabilities": self.metrics.total_vulnerabilities,
            "critical_vulnerabilities": self.metrics.critical_vulnerabilities,
            "high_vulnerabilities": self.metrics.high_vulnerabilities,
            "medium_vulnerabilities": self.metrics.medium_vulnerabilities,
            "low_vulnerabilities": self.metrics.low_vulnerabilities,
            "fixed_vulnerabilities": self.metrics.fixed_vulnerabilities,
            "false_positives": self.metrics.false_positives,
            "compliance_score": round(self.metrics.compliance_score, 2),
            "last_scan": self.metrics.last_scan.isoformat() if self.metrics.last_scan else None,
            "scan_duration": round(self.metrics.scan_duration, 2)
        }
        
    def get_vulnerabilities(self, severity: SecurityLevel = None, status: str = None) -> List[Dict[str, Any]]:
        """Get vulnerabilities with optional filtering"""
        vulnerabilities = list(self.vulnerabilities.values())
        
        if severity:
            vulnerabilities = [v for v in vulnerabilities if v.severity == severity]
        if status:
            vulnerabilities = [v for v in vulnerabilities if v.status == status]
            
        return [
            {
                "id": v.id,
                "type": v.type.value,
                "severity": v.severity.value,
                "title": v.title,
                "description": v.description,
                "file_path": v.file_path,
                "line_number": v.line_number,
                "code_snippet": v.code_snippet,
                "recommendation": v.recommendation,
                "cwe_id": v.cwe_id,
                "owasp_category": v.owasp_category,
                "detected_at": v.detected_at.isoformat(),
                "status": v.status,
                "false_positive": v.false_positive
            }
            for v in vulnerabilities
        ]
        
    def get_compliance_report(self) -> Dict[str, Any]:
        """Get compliance report"""
        standards = {}
        for compliance in self.compliance_checks:
            if compliance.standard not in standards:
                standards[compliance.standard] = {
                    "total_requirements": 0,
                    "compliant": 0,
                    "non_compliant": 0,
                    "partial": 0,
                    "requirements": []
                }
                
            standards[compliance.standard]["total_requirements"] += 1
            standards[compliance.standard][compliance.status] += 1
            standards[compliance.standard]["requirements"].append({
                "requirement": compliance.requirement,
                "status": compliance.status,
                "evidence": compliance.evidence,
                "last_checked": compliance.last_checked.isoformat()
            })
            
        return standards


# Global security auditor instance
security_auditor = SecurityAuditor()
