#!/usr/bin/env python3
"""
Security Audit Script
Comprehensive security auditing for the Opinion Market application
"""

import os
import sys
import json
import re
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SecurityIssue:
    """Represents a security issue found during audit"""
    severity: str  # critical, high, medium, low, info
    category: str  # authentication, authorization, data_protection, etc.
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    recommendation: str = ""
    cwe_id: Optional[str] = None

class SecurityAuditor:
    """Comprehensive security auditor"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: List[SecurityIssue] = []
        self.severity_colors = {
            'critical': '\033[91m',  # Red
            'high': '\033[93m',      # Yellow
            'medium': '\033[94m',    # Blue
            'low': '\033[96m',       # Cyan
            'info': '\033[92m'       # Green
        }
        self.reset_color = '\033[0m'
    
    def audit_all(self) -> List[SecurityIssue]:
        """Run all security audits"""
        print("🔒 Starting comprehensive security audit...")
        
        self.audit_dependencies()
        self.audit_code_security()
        self.audit_configuration_security()
        self.audit_database_security()
        self.audit_api_security()
        self.audit_authentication_security()
        self.audit_file_permissions()
        self.audit_secrets_management()
        
        return self.issues
    
    def audit_dependencies(self):
        """Audit Python dependencies for known vulnerabilities"""
        print("📦 Auditing dependencies...")
        
        try:
            # Check if safety is available
            result = subprocess.run(['safety', 'check', '--json'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("   ✅ No known vulnerabilities found in dependencies")
            else:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    for vuln in vulnerabilities:
                        self.issues.append(SecurityIssue(
                            severity="high",
                            category="dependencies",
                            title=f"Vulnerable dependency: {vuln.get('package_name', 'unknown')}",
                            description=f"Version {vuln.get('analyzed_version', 'unknown')} has known vulnerability: {vuln.get('advisory', 'No details')}",
                            file_path="requirements.txt",
                            recommendation=f"Update {vuln.get('package_name', 'package')} to a secure version",
                            cwe_id=vuln.get('cwe', 'CWE-1104')
                        ))
                except json.JSONDecodeError:
                    print("   ⚠️  Could not parse safety output")
        except FileNotFoundError:
            print("   ⚠️  Safety tool not found - install with: pip install safety")
    
    def audit_code_security(self):
        """Audit Python code for security issues"""
        print("🐍 Auditing Python code security...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            self._audit_python_file(file_path)
    
    def _audit_python_file(self, file_path: Path):
        """Audit a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for dangerous functions
            dangerous_patterns = [
                (r'eval\s*\(', 'critical', 'Use of eval() function', 'CWE-95'),
                (r'exec\s*\(', 'critical', 'Use of exec() function', 'CWE-95'),
                (r'__import__\s*\(', 'high', 'Use of __import__() function', 'CWE-95'),
                (r'pickle\.loads?\s*\(', 'high', 'Use of pickle.loads() - potential code injection', 'CWE-502'),
                (r'yaml\.load\s*\(', 'medium', 'Use of yaml.load() - use yaml.safe_load()', 'CWE-502'),
                (r'shell=True', 'high', 'Use of shell=True in subprocess calls', 'CWE-78'),
                (r'os\.system\s*\(', 'high', 'Use of os.system() - use subprocess instead', 'CWE-78'),
                (r'input\s*\(', 'medium', 'Use of input() in production code', 'CWE-20'),
            ]
            
            for pattern, severity, description, cwe_id in dangerous_patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        self.issues.append(SecurityIssue(
                            severity=severity,
                            category="code_security",
                            title=description,
                            description=f"Found in line {i}: {line.strip()}",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            recommendation="Review and replace with safer alternatives",
                            cwe_id=cwe_id
                        ))
            
            # Check for hardcoded secrets
            secret_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', 'high', 'Hardcoded password'),
                (r'api_key\s*=\s*["\'][^"\']+["\']', 'high', 'Hardcoded API key'),
                (r'secret\s*=\s*["\'][^"\']+["\']', 'high', 'Hardcoded secret'),
                (r'token\s*=\s*["\'][^"\']+["\']', 'high', 'Hardcoded token'),
                (r'private_key\s*=\s*["\'][^"\']+["\']', 'critical', 'Hardcoded private key'),
            ]
            
            for pattern, severity, description in secret_patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        self.issues.append(SecurityIssue(
                            severity=severity,
                            category="secrets_management",
                            title=description,
                            description=f"Found in line {i}: {line.strip()[:100]}...",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            recommendation="Use environment variables or secure secret management",
                            cwe_id="CWE-798"
                        ))
            
            # Check for SQL injection vulnerabilities
            sql_patterns = [
                (r'execute\s*\(\s*f?["\'][^"\']*%[^"\']*["\']', 'high', 'Potential SQL injection with string formatting'),
                (r'execute\s*\(\s*["\'][^"\']*\+[^"\']*["\']', 'high', 'Potential SQL injection with string concatenation'),
            ]
            
            for pattern, severity, description in sql_patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        self.issues.append(SecurityIssue(
                            severity=severity,
                            category="sql_injection",
                            title=description,
                            description=f"Found in line {i}: {line.strip()}",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            recommendation="Use parameterized queries or ORM methods",
                            cwe_id="CWE-89"
                        ))
        
        except Exception as e:
            print(f"   ⚠️  Error auditing {file_path}: {e}")
    
    def audit_configuration_security(self):
        """Audit configuration files for security issues"""
        print("⚙️  Auditing configuration security...")
        
        config_files = [
            "config/config.development.yaml",
            "config/config.production.yaml",
            "config/config.staging.yaml",
            ".env",
            ".env.local",
            ".env.production"
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                self._audit_config_file(config_path)
    
    def _audit_config_file(self, file_path: Path):
        """Audit a configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for hardcoded secrets in config files
            secret_keywords = ['password', 'secret', 'key', 'token', 'private']
            
            for i, line in enumerate(lines, 1):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in secret_keywords):
                    if '=' in line and not line.strip().startswith('#'):
                        # Check if it looks like a hardcoded value
                        if re.search(r'["\'][^"\']{8,}["\']', line):
                            self.issues.append(SecurityIssue(
                                severity="high",
                                category="configuration_security",
                                title="Potential hardcoded secret in configuration",
                                description=f"Found in line {i}: {line.strip()[:100]}...",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=i,
                                recommendation="Use environment variables or secure secret management",
                                cwe_id="CWE-798"
                            ))
        
        except Exception as e:
            print(f"   ⚠️  Error auditing config file {file_path}: {e}")
    
    def audit_database_security(self):
        """Audit database configuration and usage"""
        print("🗄️  Auditing database security...")
        
        # Check database configuration
        db_config_files = [
            "app/core/database.py",
            "app/core/config.py"
        ]
        
        for config_file in db_config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                self._audit_database_config(config_path)
    
    def _audit_database_config(self, file_path: Path):
        """Audit database configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for insecure database configurations
            insecure_patterns = [
                (r'echo=True', 'medium', 'SQLAlchemy echo=True enabled - may log sensitive data'),
                (r'pool_pre_ping=False', 'medium', 'Database connection pooling without pre-ping'),
                (r'sslmode=disable', 'high', 'SSL disabled for database connections'),
            ]
            
            for pattern, severity, description in insecure_patterns:
                if re.search(pattern, content):
                    self.issues.append(SecurityIssue(
                        severity=severity,
                        category="database_security",
                        title=description,
                        description=f"Found in {file_path.name}",
                        file_path=str(file_path.relative_to(self.project_root)),
                        recommendation="Enable secure database configurations",
                        cwe_id="CWE-319"
                    ))
        
        except Exception as e:
            print(f"   ⚠️  Error auditing database config {file_path}: {e}")
    
    def audit_api_security(self):
        """Audit API security configurations"""
        print("🌐 Auditing API security...")
        
        # Check FastAPI security configurations
        api_files = list((self.project_root / "app" / "api").rglob("*.py"))
        
        for file_path in api_files:
            self._audit_api_file(file_path)
    
    def _audit_api_file(self, file_path: Path):
        """Audit an API file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for missing authentication
            if 'router = APIRouter()' in content and 'Depends' not in content:
                self.issues.append(SecurityIssue(
                    severity="medium",
                    category="api_security",
                    title="API endpoint without authentication",
                    description="Router found without authentication dependencies",
                    file_path=str(file_path.relative_to(self.project_root)),
                    recommendation="Add authentication dependencies to protected endpoints",
                    cwe_id="CWE-306"
                ))
            
            # Check for CORS configuration
            if 'APIRouter' in content and 'CORSMiddleware' not in content:
                self.issues.append(SecurityIssue(
                    severity="low",
                    category="api_security",
                    title="Missing CORS configuration",
                    description="API router without CORS middleware",
                    file_path=str(file_path.relative_to(self.project_root)),
                    recommendation="Configure CORS middleware for web applications",
                    cwe_id="CWE-346"
                ))
        
        except Exception as e:
            print(f"   ⚠️  Error auditing API file {file_path}: {e}")
    
    def audit_authentication_security(self):
        """Audit authentication and authorization"""
        print("🔐 Auditing authentication security...")
        
        auth_files = [
            "app/api/v1/endpoints/auth.py",
            "app/core/security.py"
        ]
        
        for auth_file in auth_files:
            auth_path = self.project_root / auth_file
            if auth_path.exists():
                self._audit_auth_file(auth_path)
    
    def _audit_auth_file(self, file_path: Path):
        """Audit authentication file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for weak password hashing
            if 'bcrypt' not in content and 'argon2' not in content and 'scrypt' not in content:
                self.issues.append(SecurityIssue(
                    severity="high",
                    category="authentication",
                    title="Weak or missing password hashing",
                    description="No strong password hashing algorithm found",
                    file_path=str(file_path.relative_to(self.project_root)),
                    recommendation="Use bcrypt, argon2, or scrypt for password hashing",
                    cwe_id="CWE-916"
                ))
            
            # Check for JWT security
            if 'jwt' in content.lower():
                if 'algorithm' not in content or 'HS256' in content:
                    self.issues.append(SecurityIssue(
                        severity="medium",
                        category="authentication",
                        title="Weak JWT configuration",
                        description="JWT using weak algorithm or missing algorithm specification",
                        file_path=str(file_path.relative_to(self.project_root)),
                        recommendation="Use RS256 or ES256 for JWT signing",
                        cwe_id="CWE-327"
                    ))
        
        except Exception as e:
            print(f"   ⚠️  Error auditing auth file {file_path}: {e}")
    
    def audit_file_permissions(self):
        """Audit file permissions"""
        print("📁 Auditing file permissions...")
        
        # Check for world-writable files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    if stat.st_mode & 0o002:  # World writable
                        self.issues.append(SecurityIssue(
                            severity="medium",
                            category="file_permissions",
                            title="World-writable file",
                            description=f"File is writable by all users: {file_path.name}",
                            file_path=str(file_path.relative_to(self.project_root)),
                            recommendation="Remove world-write permissions",
                            cwe_id="CWE-732"
                        ))
                except Exception:
                    continue
    
    def audit_secrets_management(self):
        """Audit secrets management"""
        print("🔑 Auditing secrets management...")
        
        # Check for .env files in version control
        env_files = list(self.project_root.rglob(".env*"))
        for env_file in env_files:
            if not env_file.name.endswith('.example'):
                self.issues.append(SecurityIssue(
                    severity="high",
                    category="secrets_management",
                    title="Environment file in version control",
                    description=f"Environment file {env_file.name} should not be in version control",
                    file_path=str(env_file.relative_to(self.project_root)),
                    recommendation="Add to .gitignore and use .env.example instead",
                    cwe_id="CWE-798"
                ))
    
    def print_report(self):
        """Print security audit report"""
        if not self.issues:
            print(f"\n{self.severity_colors['info']}✅ No security issues found!{self.reset_color}")
            return
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in self.issues:
            if issue.severity not in issues_by_severity:
                issues_by_severity[issue.severity] = []
            issues_by_severity[issue.severity].append(issue)
        
        print(f"\n🔒 Security Audit Report")
        print("=" * 50)
        
        for severity in ['critical', 'high', 'medium', 'low', 'info']:
            if severity in issues_by_severity:
                issues = issues_by_severity[severity]
                print(f"\n{self.severity_colors[severity]}{severity.upper()}: {len(issues)} issues{self.reset_color}")
                
                for issue in issues:
                    print(f"  • {issue.title}")
                    print(f"    File: {issue.file_path}")
                    if issue.line_number:
                        print(f"    Line: {issue.line_number}")
                    print(f"    Recommendation: {issue.recommendation}")
                    if issue.cwe_id:
                        print(f"    CWE: {issue.cwe_id}")
                    print()
    
    def export_report(self, output_file: str):
        """Export security audit report to JSON"""
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_issues": len(self.issues),
                "critical": len([i for i in self.issues if i.severity == 'critical']),
                "high": len([i for i in self.issues if i.severity == 'high']),
                "medium": len([i for i in self.issues if i.severity == 'medium']),
                "low": len([i for i in self.issues if i.severity == 'low']),
                "info": len([i for i in self.issues if i.severity == 'info'])
            },
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "title": issue.title,
                    "description": issue.description,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "recommendation": issue.recommendation,
                    "cwe_id": issue.cwe_id
                }
                for issue in self.issues
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Security audit report saved to: {output_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Audit Tool")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", help="Output file for JSON report")
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor(args.project_root)
    issues = auditor.audit_all()
    
    auditor.print_report()
    
    if args.output:
        auditor.export_report(args.output)
    
    # Exit with error code if critical or high severity issues found
    critical_high_issues = [i for i in issues if i.severity in ['critical', 'high']]
    if critical_high_issues:
        sys.exit(1)

if __name__ == "__main__":
    main()
