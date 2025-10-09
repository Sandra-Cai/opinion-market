"""
Comprehensive input validation system for Opinion Market
Provides security-focused validation, sanitization, and data integrity checks
"""

import re
import html
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import ipaddress
import urllib.parse
from email_validator import validate_email, EmailNotValidError

from app.core.logging import log_security_event
from app.core.config import settings


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Validation result with detailed information"""
    is_valid: bool
    sanitized_value: Any
    errors: List[str] = None
    warnings: List[str] = None
    severity: ValidationSeverity = ValidationSeverity.INFO
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class InputValidator:
    """Comprehensive input validation system"""
    
    def __init__(self):
        self.validation_rules = {}
        self.custom_validators = {}
        self.sanitization_rules = {}
        
        # Initialize default validation rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules"""
        # SQL injection patterns
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|\/\*|\*\/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"])",
            r"(\b(OR|AND)\s+\w+\s*=\s*\w+)",
            r"(UNION\s+SELECT)",
            r"(INSERT\s+INTO)",
            r"(UPDATE\s+\w+\s+SET)",
            r"(DELETE\s+FROM)",
            r"(DROP\s+TABLE)",
            r"(CREATE\s+TABLE)",
            r"(ALTER\s+TABLE)",
            r"(EXEC\s*\()",
            r"(EXECUTE\s*\()",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>",
            r"<applet[^>]*>.*?</applet>",
            r"<meta[^>]*>",
            r"<link[^>]*>",
            r"<style[^>]*>.*?</style>",
            r"javascript:",
            r"vbscript:",
            r"on\w+\s*=",
            r"expression\s*\(",
            r"url\s*\(",
            r"@import",
            r"behavior\s*:",
            r"binding\s*:",
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"\.\.%2f",
            r"\.\.%5c",
            r"%2e%2e%2f",
            r"%2e%2e%5c",
            r"\.\.%252f",
            r"\.\.%255c",
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"[;&|`$]",
            r"\|\|",
            r"&&",
            r"`.*`",
            r"\$\(.*\)",
            r"<.*>",
            r">.*<",
        ]
    
    def validate_string(
        self,
        value: str,
        max_length: int = 1000,
        min_length: int = 0,
        allow_html: bool = False,
        allow_sql: bool = False,
        allow_xss: bool = False,
        allow_path_traversal: bool = False,
        allow_command_injection: bool = False,
        required: bool = False,
        pattern: Optional[str] = None,
        custom_validators: Optional[List[Callable]] = None
    ) -> ValidationResult:
        """Validate string input with comprehensive security checks"""
        
        result = ValidationResult(is_valid=True, sanitized_value=value)
        
        # Check if required
        if required and (not value or value.strip() == ""):
            result.is_valid = False
            result.errors.append("Field is required")
            result.severity = ValidationSeverity.ERROR
            return result
        
        # Skip validation for empty optional fields
        if not value or value.strip() == "":
            result.sanitized_value = ""
            return result
        
        # Length validation
        if len(value) > max_length:
            result.is_valid = False
            result.errors.append(f"Value exceeds maximum length of {max_length} characters")
            result.severity = ValidationSeverity.ERROR
        
        if len(value) < min_length:
            result.is_valid = False
            result.errors.append(f"Value is below minimum length of {min_length} characters")
            result.severity = ValidationSeverity.ERROR
        
        # Pattern validation
        if pattern and not re.match(pattern, value):
            result.is_valid = False
            result.errors.append("Value does not match required pattern")
            result.severity = ValidationSeverity.ERROR
        
        # Security validations
        if not allow_sql and self._contains_sql_injection(value):
            result.is_valid = False
            result.errors.append("Potential SQL injection detected")
            result.severity = ValidationSeverity.CRITICAL
            log_security_event("sql_injection_attempt", {"value": value[:100]})
        
        if not allow_xss and self._contains_xss(value):
            result.is_valid = False
            result.errors.append("Potential XSS attack detected")
            result.severity = ValidationSeverity.CRITICAL
            log_security_event("xss_attempt", {"value": value[:100]})
        
        if not allow_path_traversal and self._contains_path_traversal(value):
            result.is_valid = False
            result.errors.append("Potential path traversal attack detected")
            result.severity = ValidationSeverity.CRITICAL
            log_security_event("path_traversal_attempt", {"value": value[:100]})
        
        if not allow_command_injection and self._contains_command_injection(value):
            result.is_valid = False
            result.errors.append("Potential command injection detected")
            result.severity = ValidationSeverity.CRITICAL
            log_security_event("command_injection_attempt", {"value": value[:100]})
        
        # Custom validators
        if custom_validators:
            for validator in custom_validators:
                try:
                    validator_result = validator(value)
                    if not validator_result:
                        result.is_valid = False
                        result.errors.append("Custom validation failed")
                        result.severity = ValidationSeverity.ERROR
                except Exception as e:
                    result.is_valid = False
                    result.errors.append(f"Custom validation error: {str(e)}")
                    result.severity = ValidationSeverity.ERROR
        
        # Sanitization
        if result.is_valid:
            result.sanitized_value = self._sanitize_string(value, allow_html)
        else:
            result.sanitized_value = self._sanitize_string(value, allow_html)
        
        return result
    
    def validate_email(self, email: str, required: bool = False) -> ValidationResult:
        """Validate email address"""
        result = ValidationResult(is_valid=True, sanitized_value=email)
        
        if required and (not email or email.strip() == ""):
            result.is_valid = False
            result.errors.append("Email is required")
            result.severity = ValidationSeverity.ERROR
            return result
        
        if not email or email.strip() == "":
            result.sanitized_value = ""
            return result
        
        try:
            # Validate email format
            validated_email = validate_email(email)
            result.sanitized_value = validated_email.email
            
            # Additional security checks
            if len(email) > 254:  # RFC 5321 limit
                result.is_valid = False
                result.errors.append("Email address is too long")
                result.severity = ValidationSeverity.ERROR
            
            # Check for suspicious patterns
            if self._contains_suspicious_email_patterns(email):
                result.warnings.append("Email contains suspicious patterns")
                result.severity = ValidationSeverity.WARNING
            
        except EmailNotValidError as e:
            result.is_valid = False
            result.errors.append(f"Invalid email format: {str(e)}")
            result.severity = ValidationSeverity.ERROR
            result.sanitized_value = ""
        
        return result
    
    def validate_url(self, url: str, allowed_schemes: List[str] = None, required: bool = False) -> ValidationResult:
        """Validate URL"""
        result = ValidationResult(is_valid=True, sanitized_value=url)
        
        if allowed_schemes is None:
            allowed_schemes = ["http", "https"]
        
        if required and (not url or url.strip() == ""):
            result.is_valid = False
            result.errors.append("URL is required")
            result.severity = ValidationSeverity.ERROR
            return result
        
        if not url or url.strip() == "":
            result.sanitized_value = ""
            return result
        
        try:
            # Parse URL
            parsed = urllib.parse.urlparse(url)
            
            # Check scheme
            if parsed.scheme not in allowed_schemes:
                result.is_valid = False
                result.errors.append(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
                result.severity = ValidationSeverity.ERROR
            
            # Check for suspicious patterns
            if self._contains_suspicious_url_patterns(url):
                result.is_valid = False
                result.errors.append("URL contains suspicious patterns")
                result.severity = ValidationSeverity.CRITICAL
                log_security_event("suspicious_url", {"url": url[:100]})
            
            # Check length
            if len(url) > 2048:  # Common URL length limit
                result.is_valid = False
                result.errors.append("URL is too long")
                result.severity = ValidationSeverity.ERROR
            
            result.sanitized_value = url
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Invalid URL format: {str(e)}")
            result.severity = ValidationSeverity.ERROR
            result.sanitized_value = ""
        
        return result
    
    def validate_ip_address(self, ip: str, allowed_types: List[str] = None, required: bool = False) -> ValidationResult:
        """Validate IP address"""
        result = ValidationResult(is_valid=True, sanitized_value=ip)
        
        if allowed_types is None:
            allowed_types = ["ipv4", "ipv6"]
        
        if required and (not ip or ip.strip() == ""):
            result.is_valid = False
            result.errors.append("IP address is required")
            result.severity = ValidationSeverity.ERROR
            return result
        
        if not ip or ip.strip() == "":
            result.sanitized_value = ""
            return result
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            result.sanitized_value = str(ip_obj)
            
            # Check IP type
            if "ipv4" not in allowed_types and isinstance(ip_obj, ipaddress.IPv4Address):
                result.is_valid = False
                result.errors.append("IPv4 addresses are not allowed")
                result.severity = ValidationSeverity.ERROR
            
            if "ipv6" not in allowed_types and isinstance(ip_obj, ipaddress.IPv6Address):
                result.is_valid = False
                result.errors.append("IPv6 addresses are not allowed")
                result.severity = ValidationSeverity.ERROR
            
            # Check for private/reserved IPs
            if ip_obj.is_private:
                result.warnings.append("IP address is private")
                result.severity = ValidationSeverity.WARNING
            
            if ip_obj.is_reserved:
                result.warnings.append("IP address is reserved")
                result.severity = ValidationSeverity.WARNING
            
            if ip_obj.is_loopback:
                result.warnings.append("IP address is loopback")
                result.severity = ValidationSeverity.WARNING
            
        except ValueError as e:
            result.is_valid = False
            result.errors.append(f"Invalid IP address: {str(e)}")
            result.severity = ValidationSeverity.ERROR
            result.sanitized_value = ""
        
        return result
    
    def validate_number(
        self,
        value: Union[int, float, str],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        integer_only: bool = False,
        required: bool = False
    ) -> ValidationResult:
        """Validate numeric input"""
        result = ValidationResult(is_valid=True, sanitized_value=value)
        
        if required and (value is None or value == ""):
            result.is_valid = False
            result.errors.append("Number is required")
            result.severity = ValidationSeverity.ERROR
            return result
        
        if value is None or value == "":
            result.sanitized_value = None
            return result
        
        try:
            # Convert to number
            if integer_only:
                num_value = int(value)
            else:
                num_value = float(value)
            
            result.sanitized_value = num_value
            
            # Range validation
            if min_value is not None and num_value < min_value:
                result.is_valid = False
                result.errors.append(f"Value must be at least {min_value}")
                result.severity = ValidationSeverity.ERROR
            
            if max_value is not None and num_value > max_value:
                result.is_valid = False
                result.errors.append(f"Value must be at most {max_value}")
                result.severity = ValidationSeverity.ERROR
            
        except (ValueError, TypeError) as e:
            result.is_valid = False
            result.errors.append(f"Invalid number format: {str(e)}")
            result.severity = ValidationSeverity.ERROR
            result.sanitized_value = None
        
        return result
    
    def validate_json(self, value: str, schema: Optional[Dict] = None, required: bool = False) -> ValidationResult:
        """Validate JSON input"""
        result = ValidationResult(is_valid=True, sanitized_value=value)
        
        if required and (not value or value.strip() == ""):
            result.is_valid = False
            result.errors.append("JSON is required")
            result.severity = ValidationSeverity.ERROR
            return result
        
        if not value or value.strip() == "":
            result.sanitized_value = ""
            return result
        
        try:
            # Parse JSON
            parsed_json = json.loads(value)
            result.sanitized_value = parsed_json
            
            # Schema validation (basic)
            if schema:
                # This would integrate with a proper JSON schema validator
                result.warnings.append("Schema validation not fully implemented")
                result.severity = ValidationSeverity.WARNING
            
            # Check for suspicious JSON patterns
            if self._contains_suspicious_json_patterns(value):
                result.is_valid = False
                result.errors.append("JSON contains suspicious patterns")
                result.severity = ValidationSeverity.CRITICAL
                log_security_event("suspicious_json", {"value": value[:100]})
            
        except json.JSONDecodeError as e:
            result.is_valid = False
            result.errors.append(f"Invalid JSON format: {str(e)}")
            result.severity = ValidationSeverity.ERROR
            result.sanitized_value = ""
        
        return result
    
    def validate_uuid(self, value: str, required: bool = False) -> ValidationResult:
        """Validate UUID"""
        result = ValidationResult(is_valid=True, sanitized_value=value)
        
        if required and (not value or value.strip() == ""):
            result.is_valid = False
            result.errors.append("UUID is required")
            result.severity = ValidationSeverity.ERROR
            return result
        
        if not value or value.strip() == "":
            result.sanitized_value = ""
            return result
        
        try:
            # Validate UUID format
            uuid_obj = uuid.UUID(value)
            result.sanitized_value = str(uuid_obj)
            
        except ValueError as e:
            result.is_valid = False
            result.errors.append(f"Invalid UUID format: {str(e)}")
            result.severity = ValidationSeverity.ERROR
            result.sanitized_value = ""
        
        return result
    
    def validate_datetime(
        self,
        value: Union[str, datetime],
        format_string: Optional[str] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        required: bool = False
    ) -> ValidationResult:
        """Validate datetime input"""
        result = ValidationResult(is_valid=True, sanitized_value=value)
        
        if required and (not value or value == ""):
            result.is_valid = False
            result.errors.append("DateTime is required")
            result.severity = ValidationSeverity.ERROR
            return result
        
        if not value or value == "":
            result.sanitized_value = None
            return result
        
        try:
            # Parse datetime
            if isinstance(value, str):
                if format_string:
                    dt_value = datetime.strptime(value, format_string)
                else:
                    # Try common formats
                    for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
                        try:
                            dt_value = datetime.strptime(value, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError("Unable to parse datetime")
            else:
                dt_value = value
            
            result.sanitized_value = dt_value
            
            # Range validation
            if min_date and dt_value < min_date:
                result.is_valid = False
                result.errors.append(f"Date must be after {min_date}")
                result.severity = ValidationSeverity.ERROR
            
            if max_date and dt_value > max_date:
                result.is_valid = False
                result.errors.append(f"Date must be before {max_date}")
                result.severity = ValidationSeverity.ERROR
            
        except ValueError as e:
            result.is_valid = False
            result.errors.append(f"Invalid datetime format: {str(e)}")
            result.severity = ValidationSeverity.ERROR
            result.sanitized_value = None
        
        return result
    
    def _contains_sql_injection(self, value: str) -> bool:
        """Check for SQL injection patterns"""
        value_lower = value.lower()
        for pattern in self.sql_patterns:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return True
        return False
    
    def _contains_xss(self, value: str) -> bool:
        """Check for XSS patterns"""
        for pattern in self.xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    def _contains_path_traversal(self, value: str) -> bool:
        """Check for path traversal patterns"""
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    def _contains_command_injection(self, value: str) -> bool:
        """Check for command injection patterns"""
        for pattern in self.command_injection_patterns:
            if re.search(pattern, value):
                return True
        return False
    
    def _contains_suspicious_email_patterns(self, email: str) -> bool:
        """Check for suspicious email patterns"""
        suspicious_patterns = [
            r"\+.*@",  # Plus addressing
            r"\.{2,}",  # Multiple consecutive dots
            r"@.*@",  # Multiple @ symbols
            r"\.@",  # Dot before @
            r"@\.",  # Dot after @
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, email):
                return True
        return False
    
    def _contains_suspicious_url_patterns(self, url: str) -> bool:
        """Check for suspicious URL patterns"""
        suspicious_patterns = [
            r"javascript:",
            r"vbscript:",
            r"data:",
            r"file:",
            r"ftp:",
            r"<script",
            r"on\w+\s*=",
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False
    
    def _contains_suspicious_json_patterns(self, json_str: str) -> bool:
        """Check for suspicious JSON patterns"""
        suspicious_patterns = [
            r"__proto__",
            r"constructor",
            r"prototype",
            r"<script",
            r"javascript:",
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, json_str, re.IGNORECASE):
                return True
        return False
    
    def _sanitize_string(self, value: str, allow_html: bool = False) -> str:
        """Sanitize string input"""
        if not value:
            return ""
        
        # HTML escape if not allowing HTML
        if not allow_html:
            value = html.escape(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Normalize whitespace
        value = ' '.join(value.split())
        
        return value.strip()
    
    def validate_dict(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
        required_fields: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate dictionary against schema"""
        result = ValidationResult(is_valid=True, sanitized_value={})
        
        if required_fields is None:
            required_fields = []
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                result.is_valid = False
                result.errors.append(f"Required field '{field}' is missing")
                result.severity = ValidationSeverity.ERROR
        
        # Validate each field
        for field, value in data.items():
            if field in schema:
                field_schema = schema[field]
                field_result = self._validate_field(value, field_schema)
                
                if not field_result.is_valid:
                    result.is_valid = False
                    result.errors.extend([f"{field}: {error}" for error in field_result.errors])
                    result.severity = max(result.severity, field_result.severity)
                
                result.sanitized_value[field] = field_result.sanitized_value
            else:
                # Field not in schema, keep as is
                result.sanitized_value[field] = value
        
        return result
    
    def _validate_field(self, value: Any, schema: Dict[str, Any]) -> ValidationResult:
        """Validate a single field against its schema"""
        field_type = schema.get("type", "string")
        
        if field_type == "string":
            return self.validate_string(
                str(value),
                max_length=schema.get("max_length", 1000),
                min_length=schema.get("min_length", 0),
                allow_html=schema.get("allow_html", False),
                required=schema.get("required", False),
                pattern=schema.get("pattern")
            )
        elif field_type == "email":
            return self.validate_email(str(value), required=schema.get("required", False))
        elif field_type == "url":
            return self.validate_url(
                str(value),
                allowed_schemes=schema.get("allowed_schemes"),
                required=schema.get("required", False)
            )
        elif field_type == "number":
            return self.validate_number(
                value,
                min_value=schema.get("min_value"),
                max_value=schema.get("max_value"),
                integer_only=schema.get("integer_only", False),
                required=schema.get("required", False)
            )
        elif field_type == "datetime":
            return self.validate_datetime(
                value,
                format_string=schema.get("format"),
                min_date=schema.get("min_date"),
                max_date=schema.get("max_date"),
                required=schema.get("required", False)
            )
        else:
            # Unknown type, return as is
            return ValidationResult(is_valid=True, sanitized_value=value)


# Global input validator instance
input_validator = InputValidator()
