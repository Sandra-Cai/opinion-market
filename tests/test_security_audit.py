"""
Security Audit Test Suite
Tests the comprehensive security audit system
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

from app.core.security_audit import (
    security_auditor, 
    SecurityLevel, 
    VulnerabilityType, 
    SecurityVulnerability,
    SecurityCompliance
)


class TestSecurityAuditor:
    """Test security audit functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        # Clear existing data
        security_auditor.vulnerabilities.clear()
        security_auditor.compliance_checks.clear()
        security_auditor.scan_in_progress = False
    
    def test_security_auditor_initialization(self):
        """Test security auditor initialization"""
        assert len(security_auditor.vulnerabilities) == 0
        assert len(security_auditor.compliance_checks) == 0
        assert not security_auditor.scan_in_progress
        assert len(security_auditor.scan_patterns) > 0
        assert len(security_auditor.compliance_standards) > 0
    
    def test_vulnerability_id_generation(self):
        """Test vulnerability ID generation"""
        vuln_id1 = security_auditor._generate_vuln_id("test.py", 10, VulnerabilityType.SQL_INJECTION)
        vuln_id2 = security_auditor._generate_vuln_id("test.py", 10, VulnerabilityType.SQL_INJECTION)
        vuln_id3 = security_auditor._generate_vuln_id("test.py", 11, VulnerabilityType.SQL_INJECTION)
        
        # Same file, line, and type should generate same ID
        assert vuln_id1 == vuln_id2
        
        # Different line should generate different ID
        assert vuln_id1 != vuln_id3
        
        # ID should be 12 characters long
        assert len(vuln_id1) == 12
    
    def test_severity_mapping(self):
        """Test vulnerability severity mapping"""
        assert security_auditor._get_severity(VulnerabilityType.SQL_INJECTION) == SecurityLevel.CRITICAL
        assert security_auditor._get_severity(VulnerabilityType.XSS) == SecurityLevel.HIGH
        assert security_auditor._get_severity(VulnerabilityType.PATH_TRAVERSAL) == SecurityLevel.HIGH
        assert security_auditor._get_severity(VulnerabilityType.COMMAND_INJECTION) == SecurityLevel.CRITICAL
        assert security_auditor._get_severity(VulnerabilityType.WEAK_AUTHENTICATION) == SecurityLevel.HIGH
        assert security_auditor._get_severity(VulnerabilityType.SENSITIVE_DATA_EXPOSURE) == SecurityLevel.MEDIUM
    
    def test_vulnerability_titles(self):
        """Test vulnerability title generation"""
        assert "SQL Injection" in security_auditor._get_vuln_title(VulnerabilityType.SQL_INJECTION)
        assert "Cross-Site Scripting" in security_auditor._get_vuln_title(VulnerabilityType.XSS)
        assert "Path Traversal" in security_auditor._get_vuln_title(VulnerabilityType.PATH_TRAVERSAL)
        assert "Command Injection" in security_auditor._get_vuln_title(VulnerabilityType.COMMAND_INJECTION)
    
    def test_vulnerability_descriptions(self):
        """Test vulnerability description generation"""
        sql_desc = security_auditor._get_vuln_description(VulnerabilityType.SQL_INJECTION)
        assert "SQL injection" in sql_desc.lower()
        assert "concatenated" in sql_desc.lower()
        
        xss_desc = security_auditor._get_vuln_description(VulnerabilityType.XSS)
        assert "cross-site scripting" in xss_desc.lower()
        assert "sanitization" in xss_desc.lower()
    
    def test_recommendations(self):
        """Test remediation recommendations"""
        sql_rec = security_auditor._get_recommendation(VulnerabilityType.SQL_INJECTION)
        assert "parameterized queries" in sql_rec.lower()
        assert "prepared statements" in sql_rec.lower()
        
        xss_rec = security_auditor._get_recommendation(VulnerabilityType.XSS)
        assert "sanitize" in xss_rec.lower()
        assert "validate" in xss_rec.lower()
    
    def test_cwe_mapping(self):
        """Test CWE ID mapping"""
        assert security_auditor._get_cwe_id(VulnerabilityType.SQL_INJECTION) == "CWE-89"
        assert security_auditor._get_cwe_id(VulnerabilityType.XSS) == "CWE-79"
        assert security_auditor._get_cwe_id(VulnerabilityType.PATH_TRAVERSAL) == "CWE-22"
        assert security_auditor._get_cwe_id(VulnerabilityType.COMMAND_INJECTION) == "CWE-78"
    
    def test_owasp_mapping(self):
        """Test OWASP Top 10 mapping"""
        assert "A03:2021" in security_auditor._get_owasp_category(VulnerabilityType.SQL_INJECTION)
        assert "A03:2021" in security_auditor._get_owasp_category(VulnerabilityType.XSS)
        assert "A01:2021" in security_auditor._get_owasp_category(VulnerabilityType.PATH_TRAVERSAL)
        assert "A07:2021" in security_auditor._get_owasp_category(VulnerabilityType.WEAK_AUTHENTICATION)
    
    @pytest.mark.asyncio
    async def test_scan_file_sql_injection(self):
        """Test scanning file for SQL injection vulnerabilities"""
        # Create temporary file with SQL injection vulnerability
        vulnerable_code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(vulnerable_code)
            temp_file = f.name
        
        try:
            await security_auditor._scan_file(temp_file)
            
            # Should detect SQL injection vulnerability
            assert len(security_auditor.vulnerabilities) > 0
            
            vuln = list(security_auditor.vulnerabilities.values())[0]
            assert vuln.type == VulnerabilityType.SQL_INJECTION
            assert vuln.severity == SecurityLevel.CRITICAL
            assert "SELECT * FROM users" in vuln.code_snippet
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_scan_file_xss(self):
        """Test scanning file for XSS vulnerabilities"""
        # Create temporary file with XSS vulnerability
        vulnerable_code = """
def render_template(template_string):
    return render_template_string(template_string)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(vulnerable_code)
            temp_file = f.name
        
        try:
            await security_auditor._scan_file(temp_file)
            
            # Should detect XSS vulnerability
            assert len(security_auditor.vulnerabilities) > 0
            
            vuln = list(security_auditor.vulnerabilities.values())[0]
            assert vuln.type == VulnerabilityType.XSS
            assert vuln.severity == SecurityLevel.HIGH
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_scan_file_path_traversal(self):
        """Test scanning file for path traversal vulnerabilities"""
        # Create temporary file with path traversal vulnerability
        vulnerable_code = """
def read_file(filename):
    with open(f"../{filename}", 'r') as f:
        return f.read()
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(vulnerable_code)
            temp_file = f.name
        
        try:
            await security_auditor._scan_file(temp_file)
            
            # Should detect path traversal vulnerability
            assert len(security_auditor.vulnerabilities) > 0
            
            vuln = list(security_auditor.vulnerabilities.values())[0]
            assert vuln.type == VulnerabilityType.PATH_TRAVERSAL
            assert vuln.severity == SecurityLevel.HIGH
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_scan_file_command_injection(self):
        """Test scanning file for command injection vulnerabilities"""
        # Create temporary file with command injection vulnerability
        vulnerable_code = """
def run_command(cmd):
    return os.system(cmd)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(vulnerable_code)
            temp_file = f.name
        
        try:
            await security_auditor._scan_file(temp_file)
            
            # Should detect command injection vulnerability
            assert len(security_auditor.vulnerabilities) > 0
            
            vuln = list(security_auditor.vulnerabilities.values())[0]
            assert vuln.type == VulnerabilityType.COMMAND_INJECTION
            assert vuln.severity == SecurityLevel.CRITICAL
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_scan_file_weak_authentication(self):
        """Test scanning file for weak authentication"""
        # Create temporary file with weak authentication
        vulnerable_code = """
SECRET_KEY = "test"
DEBUG = True
password = "123456"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(vulnerable_code)
            temp_file = f.name
        
        try:
            await security_auditor._scan_file(temp_file)
            
            # Should detect weak authentication vulnerabilities
            assert len(security_auditor.vulnerabilities) > 0
            
            # Check for weak password
            weak_password_vuln = None
            for vuln in security_auditor.vulnerabilities.values():
                if vuln.type == VulnerabilityType.WEAK_AUTHENTICATION:
                    weak_password_vuln = vuln
                    break
            
            assert weak_password_vuln is not None
            assert weak_password_vuln.severity == SecurityLevel.HIGH
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_scan_file_sensitive_data_exposure(self):
        """Test scanning file for sensitive data exposure"""
        # Create temporary file with sensitive data exposure
        vulnerable_code = """
def login(username, password):
    print(f"User {username} logged in with password {password}")
    logger.info(f"Password: {password}")
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(vulnerable_code)
            temp_file = f.name
        
        try:
            await security_auditor._scan_file(temp_file)
            
            # Should detect sensitive data exposure
            assert len(security_auditor.vulnerabilities) > 0
            
            vuln = list(security_auditor.vulnerabilities.values())[0]
            assert vuln.type == VulnerabilityType.SENSITIVE_DATA_EXPOSURE
            assert vuln.severity == SecurityLevel.MEDIUM
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_scan_dependencies(self):
        """Test dependency vulnerability scanning"""
        # Mock subprocess.run to simulate safety check
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '[{"package": "requests", "advisory": "Test vulnerability", "safe_version": "2.28.0", "cve": "CVE-2023-1234"}]'
        
        with patch('subprocess.run', return_value=mock_result), \
             patch('os.path.exists', return_value=True):
            
            await security_auditor._scan_dependencies()
            
            # Should detect dependency vulnerability
            assert len(security_auditor.vulnerabilities) > 0
            
            vuln = list(security_auditor.vulnerabilities.values())[0]
            assert vuln.type == VulnerabilityType.SECURITY_MISCONFIGURATION
            assert vuln.severity == SecurityLevel.HIGH
            assert "requests" in vuln.title
    
    @pytest.mark.asyncio
    async def test_check_compliance(self):
        """Test compliance checking"""
        await security_auditor._check_compliance()
        
        # Should have compliance checks for all standards
        assert len(security_auditor.compliance_checks) > 0
        
        # Check that we have checks for OWASP and NIST
        standards = set(check.standard for check in security_auditor.compliance_checks)
        assert "OWASP_TOP_10" in standards
        assert "NIST_CYBERSECURITY_FRAMEWORK" in standards
    
    @pytest.mark.asyncio
    async def test_comprehensive_scan(self):
        """Test comprehensive security scan"""
        # Create temporary directory with vulnerable files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create vulnerable Python file
            vulnerable_file = os.path.join(temp_dir, "vulnerable.py")
            with open(vulnerable_file, 'w') as f:
                f.write('query = f"SELECT * FROM users WHERE id = {user_id}"\ncursor.execute(query)')
            
            # Run comprehensive scan
            result = await security_auditor.run_comprehensive_scan([temp_dir])
            
            # Should complete successfully
            assert result["status"] == "completed"
            assert "vulnerabilities_found" in result
            assert "compliance_checks" in result
            assert "scan_duration" in result
            
            # Should find vulnerabilities
            assert result["vulnerabilities_found"] > 0
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        # Add test vulnerabilities
        vuln1 = SecurityVulnerability(
            id="test1", type=VulnerabilityType.SQL_INJECTION, severity=SecurityLevel.CRITICAL,
            title="Test 1", description="Test", file_path="test.py", line_number=1,
            code_snippet="test", recommendation="test"
        )
        vuln2 = SecurityVulnerability(
            id="test2", type=VulnerabilityType.XSS, severity=SecurityLevel.HIGH,
            title="Test 2", description="Test", file_path="test.py", line_number=2,
            code_snippet="test", recommendation="test"
        )
        vuln3 = SecurityVulnerability(
            id="test3", type=VulnerabilityType.SENSITIVE_DATA_EXPOSURE, severity=SecurityLevel.MEDIUM,
            title="Test 3", description="Test", file_path="test.py", line_number=3,
            code_snippet="test", recommendation="test", status="fixed"
        )
        
        security_auditor.vulnerabilities = {
            "test1": vuln1,
            "test2": vuln2,
            "test3": vuln3
        }
        
        # Add test compliance checks
        compliance1 = SecurityCompliance("OWASP_TOP_10", "Test requirement", "compliant")
        compliance2 = SecurityCompliance("OWASP_TOP_10", "Test requirement 2", "non_compliant")
        
        security_auditor.compliance_checks = [compliance1, compliance2]
        
        # Calculate metrics
        security_auditor._calculate_metrics()
        
        # Check metrics
        assert security_auditor.metrics.total_vulnerabilities == 3
        assert security_auditor.metrics.critical_vulnerabilities == 1
        assert security_auditor.metrics.high_vulnerabilities == 1
        assert security_auditor.metrics.medium_vulnerabilities == 1
        assert security_auditor.metrics.fixed_vulnerabilities == 1
        assert security_auditor.metrics.compliance_score == 50.0  # 1 out of 2 compliant
    
    def test_get_vulnerabilities_filtering(self):
        """Test vulnerability filtering"""
        # Add test vulnerabilities
        vuln1 = SecurityVulnerability(
            id="test1", type=VulnerabilityType.SQL_INJECTION, severity=SecurityLevel.CRITICAL,
            title="Test 1", description="Test", file_path="test.py", line_number=1,
            code_snippet="test", recommendation="test"
        )
        vuln2 = SecurityVulnerability(
            id="test2", type=VulnerabilityType.XSS, severity=SecurityLevel.HIGH,
            title="Test 2", description="Test", file_path="test.py", line_number=2,
            code_snippet="test", recommendation="test", status="fixed"
        )
        
        security_auditor.vulnerabilities = {"test1": vuln1, "test2": vuln2}
        
        # Test filtering by severity
        critical_vulns = security_auditor.get_vulnerabilities(severity=SecurityLevel.CRITICAL)
        assert len(critical_vulns) == 1
        assert critical_vulns[0]["severity"] == "critical"
        
        # Test filtering by status
        fixed_vulns = security_auditor.get_vulnerabilities(status="fixed")
        assert len(fixed_vulns) == 1
        assert fixed_vulns[0]["status"] == "fixed"
        
        # Test no filtering
        all_vulns = security_auditor.get_vulnerabilities()
        assert len(all_vulns) == 2
    
    def test_get_compliance_report(self):
        """Test compliance report generation"""
        # Add test compliance checks
        compliance1 = SecurityCompliance("OWASP_TOP_10", "A01:2021 – Broken Access Control", "compliant")
        compliance2 = SecurityCompliance("OWASP_TOP_10", "A02:2021 – Cryptographic Failures", "non_compliant")
        compliance3 = SecurityCompliance("NIST_CYBERSECURITY_FRAMEWORK", "Identify: Asset Management", "partial")
        
        security_auditor.compliance_checks = [compliance1, compliance2, compliance3]
        
        # Get compliance report
        report = security_auditor.get_compliance_report()
        
        # Should have both standards
        assert "OWASP_TOP_10" in report
        assert "NIST_CYBERSECURITY_FRAMEWORK" in report
        
        # Check OWASP data
        owasp_data = report["OWASP_TOP_10"]
        assert owasp_data["total_requirements"] == 2
        assert owasp_data["compliant"] == 1
        assert owasp_data["non_compliant"] == 1
        assert owasp_data["partial"] == 0
        
        # Check NIST data
        nist_data = report["NIST_CYBERSECURITY_FRAMEWORK"]
        assert nist_data["total_requirements"] == 1
        assert nist_data["compliant"] == 0
        assert nist_data["non_compliant"] == 0
        assert nist_data["partial"] == 1


class TestSecurityAuditIntegration:
    """Integration tests for security audit system"""
    
    @pytest.mark.asyncio
    async def test_security_audit_integration(self):
        """Test integration of security audit components"""
        # Create temporary directory with multiple vulnerable files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different vulnerability types
            files = {
                "sql_injection.py": 'query = f"SELECT * FROM users WHERE id = {user_id}"',
                "xss.py": 'return render_template_string(user_input)',
                "path_traversal.py": 'with open(f"../{filename}", "r") as f:',
                "command_injection.py": 'os.system(user_command)',
                "weak_auth.py": 'password = "123456"\nDEBUG = True',
                "data_exposure.py": 'print(f"Password: {password}")'
            }
            
            for filename, content in files.items():
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'w') as f:
                    f.write(content)
            
            # Run comprehensive scan
            result = await security_auditor.run_comprehensive_scan([temp_dir])
            
            # Should complete successfully
            assert result["status"] == "completed"
            assert result["vulnerabilities_found"] > 0
            
            # Should have found multiple vulnerability types
            vuln_types = set(vuln.type for vuln in security_auditor.vulnerabilities.values())
            assert len(vuln_types) > 1
            
            # Should have compliance checks
            assert result["compliance_checks"] > 0
            
            # Should have metrics
            metrics = result["metrics"]
            assert "total_vulnerabilities" in metrics
            assert "critical_vulnerabilities" in metrics
            assert "compliance_score" in metrics
