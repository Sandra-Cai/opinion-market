"""
Unit tests for validation module
"""

import pytest
from datetime import datetime, timedelta
import json
import uuid

from app.core.validation import (
    input_validator,
    ValidationResult,
    ValidationSeverity
)


class TestStringValidation:
    """Test cases for string validation"""
    
    def test_validate_string_basic(self):
        """Test basic string validation"""
        result = input_validator.validate_string("test string")
        
        assert result.is_valid is True
        assert result.sanitized_value == "test string"
        assert len(result.errors) == 0
    
    def test_validate_string_required(self):
        """Test required string validation"""
        # Empty string when required
        result = input_validator.validate_string("", required=True)
        assert result.is_valid is False
        assert "Field is required" in result.errors
        
        # Valid string when required
        result = input_validator.validate_string("valid", required=True)
        assert result.is_valid is True
    
    def test_validate_string_length_limits(self):
        """Test string length validation"""
        # Too long
        long_string = "a" * 1001
        result = input_validator.validate_string(long_string, max_length=1000)
        assert result.is_valid is False
        assert "exceeds maximum length" in result.errors[0]
        
        # Too short
        result = input_validator.validate_string("ab", min_length=3)
        assert result.is_valid is False
        assert "below minimum length" in result.errors[0]
        
        # Valid length
        result = input_validator.validate_string("valid", min_length=3, max_length=10)
        assert result.is_valid is True
    
    def test_validate_string_pattern(self):
        """Test string pattern validation"""
        # Valid pattern
        result = input_validator.validate_string("test123", pattern=r"^[a-z0-9]+$")
        assert result.is_valid is True
        
        # Invalid pattern
        result = input_validator.validate_string("test-123", pattern=r"^[a-z0-9]+$")
        assert result.is_valid is False
        assert "does not match required pattern" in result.errors[0]
    
    def test_validate_string_sql_injection(self):
        """Test SQL injection detection"""
        # SQL injection attempt
        sql_injection = "'; DROP TABLE users; --"
        result = input_validator.validate_string(sql_injection)
        assert result.is_valid is False
        assert "SQL injection detected" in result.errors[0]
        assert result.severity == ValidationSeverity.CRITICAL
        
        # Allow SQL (for testing)
        result = input_validator.validate_string(sql_injection, allow_sql=True)
        assert result.is_valid is True
    
    def test_validate_string_xss(self):
        """Test XSS detection"""
        # XSS attempt
        xss_attempt = "<script>alert('xss')</script>"
        result = input_validator.validate_string(xss_attempt)
        assert result.is_valid is False
        assert "XSS attack detected" in result.errors[0]
        assert result.severity == ValidationSeverity.CRITICAL
        
        # Allow XSS (for testing)
        result = input_validator.validate_string(xss_attempt, allow_xss=True)
        assert result.is_valid is True
    
    def test_validate_string_path_traversal(self):
        """Test path traversal detection"""
        # Path traversal attempt
        path_traversal = "../../../etc/passwd"
        result = input_validator.validate_string(path_traversal)
        assert result.is_valid is False
        assert "path traversal attack detected" in result.errors[0]
        assert result.severity == ValidationSeverity.CRITICAL
        
        # Allow path traversal (for testing)
        result = input_validator.validate_string(path_traversal, allow_path_traversal=True)
        assert result.is_valid is True
    
    def test_validate_string_command_injection(self):
        """Test command injection detection"""
        # Command injection attempt
        command_injection = "test; rm -rf /"
        result = input_validator.validate_string(command_injection)
        assert result.is_valid is False
        assert "command injection detected" in result.errors[0]
        assert result.severity == ValidationSeverity.CRITICAL
        
        # Allow command injection (for testing)
        result = input_validator.validate_string(command_injection, allow_command_injection=True)
        assert result.is_valid is True
    
    def test_validate_string_sanitization(self):
        """Test string sanitization"""
        # Test HTML escaping
        html_string = "<script>alert('test')</script>"
        result = input_validator.validate_string(html_string, allow_html=False)
        assert "&lt;script&gt;" in result.sanitized_value
        
        # Test null byte removal
        null_string = "test\x00string"
        result = input_validator.validate_string(null_string)
        assert "\x00" not in result.sanitized_value
        
        # Test whitespace normalization
        whitespace_string = "  test   string  "
        result = input_validator.validate_string(whitespace_string)
        assert result.sanitized_value == "test string"


class TestEmailValidation:
    """Test cases for email validation"""
    
    def test_validate_email_valid(self):
        """Test valid email validation"""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org"
        ]
        
        for email in valid_emails:
            result = input_validator.validate_email(email)
            assert result.is_valid is True
            assert result.sanitized_value == email
    
    def test_validate_email_invalid(self):
        """Test invalid email validation"""
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "test@",
            "test..test@example.com",
            "test@.com"
        ]
        
        for email in invalid_emails:
            result = input_validator.validate_email(email)
            assert result.is_valid is False
            assert len(result.errors) > 0
    
    def test_validate_email_required(self):
        """Test required email validation"""
        # Empty email when required
        result = input_validator.validate_email("", required=True)
        assert result.is_valid is False
        assert "Email is required" in result.errors
        
        # Valid email when required
        result = input_validator.validate_email("test@example.com", required=True)
        assert result.is_valid is True
    
    def test_validate_email_suspicious_patterns(self):
        """Test suspicious email pattern detection"""
        suspicious_emails = [
            "test+spam@example.com",  # Plus addressing
            "test..test@example.com",  # Multiple dots
            "test@@example.com",  # Multiple @ symbols
        ]
        
        for email in suspicious_emails:
            result = input_validator.validate_email(email)
            # Should still be valid but with warnings
            assert result.is_valid is True
            assert len(result.warnings) > 0


class TestURLValidation:
    """Test cases for URL validation"""
    
    def test_validate_url_valid(self):
        """Test valid URL validation"""
        valid_urls = [
            "https://example.com",
            "http://example.com/path",
            "https://subdomain.example.com/path?param=value"
        ]
        
        for url in valid_urls:
            result = input_validator.validate_url(url)
            assert result.is_valid is True
            assert result.sanitized_value == url
    
    def test_validate_url_invalid(self):
        """Test invalid URL validation"""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Not in allowed schemes
            "javascript:alert('xss')",  # Suspicious scheme
        ]
        
        for url in invalid_urls:
            result = input_validator.validate_url(url)
            assert result.is_valid is False
            assert len(result.errors) > 0
    
    def test_validate_url_allowed_schemes(self):
        """Test URL validation with custom allowed schemes"""
        # Only allow HTTPS
        result = input_validator.validate_url("https://example.com", allowed_schemes=["https"])
        assert result.is_valid is True
        
        # HTTP not allowed
        result = input_validator.validate_url("http://example.com", allowed_schemes=["https"])
        assert result.is_valid is False
        assert "scheme must be one of" in result.errors[0]
    
    def test_validate_url_suspicious_patterns(self):
        """Test suspicious URL pattern detection"""
        suspicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd"
        ]
        
        for url in suspicious_urls:
            result = input_validator.validate_url(url)
            assert result.is_valid is False
            assert "suspicious patterns" in result.errors[0]
            assert result.severity == ValidationSeverity.CRITICAL


class TestIPAddressValidation:
    """Test cases for IP address validation"""
    
    def test_validate_ip_address_valid(self):
        """Test valid IP address validation"""
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        ]
        
        for ip in valid_ips:
            result = input_validator.validate_ip_address(ip)
            assert result.is_valid is True
            assert result.sanitized_value == ip
    
    def test_validate_ip_address_invalid(self):
        """Test invalid IP address validation"""
        invalid_ips = [
            "not-an-ip",
            "256.256.256.256",
            "192.168.1.1.1"
        ]
        
        for ip in invalid_ips:
            result = input_validator.validate_ip_address(ip)
            assert result.is_valid is False
            assert len(result.errors) > 0
    
    def test_validate_ip_address_types(self):
        """Test IP address type validation"""
        # IPv4 only
        result = input_validator.validate_ip_address("192.168.1.1", allowed_types=["ipv4"])
        assert result.is_valid is True
        
        result = input_validator.validate_ip_address("2001:0db8::1", allowed_types=["ipv4"])
        assert result.is_valid is False
        assert "IPv6 addresses are not allowed" in result.errors[0]
        
        # IPv6 only
        result = input_validator.validate_ip_address("2001:0db8::1", allowed_types=["ipv6"])
        assert result.is_valid is True
        
        result = input_validator.validate_ip_address("192.168.1.1", allowed_types=["ipv6"])
        assert result.is_valid is False
        assert "IPv4 addresses are not allowed" in result.errors[0]
    
    def test_validate_ip_address_warnings(self):
        """Test IP address warnings"""
        # Private IP
        result = input_validator.validate_ip_address("192.168.1.1")
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "private" in result.warnings[0]
        
        # Loopback IP
        result = input_validator.validate_ip_address("127.0.0.1")
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "loopback" in result.warnings[0]


class TestNumberValidation:
    """Test cases for number validation"""
    
    def test_validate_number_valid(self):
        """Test valid number validation"""
        valid_numbers = [1, 1.5, "123", "45.67"]
        
        for num in valid_numbers:
            result = input_validator.validate_number(num)
            assert result.is_valid is True
            assert isinstance(result.sanitized_value, (int, float))
    
    def test_validate_number_invalid(self):
        """Test invalid number validation"""
        invalid_numbers = ["not-a-number", "abc", None]
        
        for num in invalid_numbers:
            result = input_validator.validate_number(num)
            assert result.is_valid is False
            assert len(result.errors) > 0
    
    def test_validate_number_range(self):
        """Test number range validation"""
        # Within range
        result = input_validator.validate_number(5, min_value=1, max_value=10)
        assert result.is_valid is True
        
        # Below minimum
        result = input_validator.validate_number(0, min_value=1, max_value=10)
        assert result.is_valid is False
        assert "at least 1" in result.errors[0]
        
        # Above maximum
        result = input_validator.validate_number(15, min_value=1, max_value=10)
        assert result.is_valid is False
        assert "at most 10" in result.errors[0]
    
    def test_validate_number_integer_only(self):
        """Test integer-only validation"""
        # Valid integer
        result = input_validator.validate_number(5, integer_only=True)
        assert result.is_valid is True
        assert isinstance(result.sanitized_value, int)
        
        # Float when integer required
        result = input_validator.validate_number(5.5, integer_only=True)
        assert result.is_valid is False
        assert "Invalid number format" in result.errors[0]
    
    def test_validate_number_required(self):
        """Test required number validation"""
        # Empty when required
        result = input_validator.validate_number(None, required=True)
        assert result.is_valid is False
        assert "Number is required" in result.errors
        
        # Valid number when required
        result = input_validator.validate_number(5, required=True)
        assert result.is_valid is True


class TestJSONValidation:
    """Test cases for JSON validation"""
    
    def test_validate_json_valid(self):
        """Test valid JSON validation"""
        valid_jsons = [
            '{"key": "value"}',
            '[1, 2, 3]',
            '{"nested": {"key": "value"}}'
        ]
        
        for json_str in valid_jsons:
            result = input_validator.validate_json(json_str)
            assert result.is_valid is True
            assert isinstance(result.sanitized_value, (dict, list))
    
    def test_validate_json_invalid(self):
        """Test invalid JSON validation"""
        invalid_jsons = [
            '{"key": "value"',  # Missing closing brace
            '{key: "value"}',  # Unquoted key
            'not json'
        ]
        
        for json_str in invalid_jsons:
            result = input_validator.validate_json(json_str)
            assert result.is_valid is False
            assert "Invalid JSON format" in result.errors[0]
    
    def test_validate_json_required(self):
        """Test required JSON validation"""
        # Empty when required
        result = input_validator.validate_json("", required=True)
        assert result.is_valid is False
        assert "JSON is required" in result.errors
        
        # Valid JSON when required
        result = input_validator.validate_json('{"key": "value"}', required=True)
        assert result.is_valid is True
    
    def test_validate_json_suspicious_patterns(self):
        """Test suspicious JSON pattern detection"""
        suspicious_jsons = [
            '{"__proto__": {"polluted": true}}',
            '{"constructor": {"prototype": {"polluted": true}}}',
            '{"<script>alert(\'xss\')</script>": "value"}'
        ]
        
        for json_str in suspicious_jsons:
            result = input_validator.validate_json(json_str)
            assert result.is_valid is False
            assert "suspicious patterns" in result.errors[0]
            assert result.severity == ValidationSeverity.CRITICAL


class TestUUIDValidation:
    """Test cases for UUID validation"""
    
    def test_validate_uuid_valid(self):
        """Test valid UUID validation"""
        valid_uuids = [
            str(uuid.uuid4()),
            str(uuid.uuid1()),
            "550e8400-e29b-41d4-a716-446655440000"
        ]
        
        for uuid_str in valid_uuids:
            result = input_validator.validate_uuid(uuid_str)
            assert result.is_valid is True
            assert result.sanitized_value == uuid_str
    
    def test_validate_uuid_invalid(self):
        """Test invalid UUID validation"""
        invalid_uuids = [
            "not-a-uuid",
            "550e8400-e29b-41d4-a716",
            "550e8400-e29b-41d4-a716-44665544000g"  # Invalid character
        ]
        
        for uuid_str in invalid_uuids:
            result = input_validator.validate_uuid(uuid_str)
            assert result.is_valid is False
            assert "Invalid UUID format" in result.errors[0]
    
    def test_validate_uuid_required(self):
        """Test required UUID validation"""
        # Empty when required
        result = input_validator.validate_uuid("", required=True)
        assert result.is_valid is False
        assert "UUID is required" in result.errors
        
        # Valid UUID when required
        result = input_validator.validate_uuid(str(uuid.uuid4()), required=True)
        assert result.is_valid is True


class TestDateTimeValidation:
    """Test cases for datetime validation"""
    
    def test_validate_datetime_valid(self):
        """Test valid datetime validation"""
        valid_datetimes = [
            "2023-01-01",
            "2023-01-01 12:00:00",
            "2023-01-01T12:00:00",
            "2023-01-01T12:00:00Z"
        ]
        
        for dt_str in valid_datetimes:
            result = input_validator.validate_datetime(dt_str)
            assert result.is_valid is True
            assert isinstance(result.sanitized_value, datetime)
    
    def test_validate_datetime_invalid(self):
        """Test invalid datetime validation"""
        invalid_datetimes = [
            "not-a-date",
            "2023-13-01",  # Invalid month
            "2023-01-32"   # Invalid day
        ]
        
        for dt_str in invalid_datetimes:
            result = input_validator.validate_datetime(dt_str)
            assert result.is_valid is False
            assert "Invalid datetime format" in result.errors[0]
    
    def test_validate_datetime_range(self):
        """Test datetime range validation"""
        now = datetime.utcnow()
        past = now - timedelta(days=1)
        future = now + timedelta(days=1)
        
        # Within range
        result = input_validator.validate_datetime(
            now.isoformat(),
            min_date=past,
            max_date=future
        )
        assert result.is_valid is True
        
        # Before minimum
        result = input_validator.validate_datetime(
            past.isoformat(),
            min_date=now,
            max_date=future
        )
        assert result.is_valid is False
        assert "after" in result.errors[0]
        
        # After maximum
        result = input_validator.validate_datetime(
            future.isoformat(),
            min_date=past,
            max_date=now
        )
        assert result.is_valid is False
        assert "before" in result.errors[0]
    
    def test_validate_datetime_format(self):
        """Test datetime format validation"""
        # Custom format
        result = input_validator.validate_datetime(
            "01/01/2023",
            format_string="%m/%d/%Y"
        )
        assert result.is_valid is True
        
        # Wrong format
        result = input_validator.validate_datetime(
            "2023-01-01",
            format_string="%m/%d/%Y"
        )
        assert result.is_valid is False


class TestDictValidation:
    """Test cases for dictionary validation"""
    
    def test_validate_dict_valid(self):
        """Test valid dictionary validation"""
        schema = {
            "name": {"type": "string", "required": True},
            "age": {"type": "number", "required": False}
        }
        
        data = {"name": "John", "age": 30}
        result = input_validator.validate_dict(data, schema, required_fields=["name"])
        
        assert result.is_valid is True
        assert result.sanitized_value["name"] == "John"
        assert result.sanitized_value["age"] == 30
    
    def test_validate_dict_missing_required(self):
        """Test dictionary validation with missing required fields"""
        schema = {
            "name": {"type": "string", "required": True},
            "age": {"type": "number", "required": False}
        }
        
        data = {"age": 30}  # Missing required 'name'
        result = input_validator.validate_dict(data, schema, required_fields=["name"])
        
        assert result.is_valid is False
        assert "Required field 'name' is missing" in result.errors[0]
    
    def test_validate_dict_invalid_field(self):
        """Test dictionary validation with invalid field values"""
        schema = {
            "age": {"type": "number", "required": True}
        }
        
        data = {"age": "not-a-number"}
        result = input_validator.validate_dict(data, schema, required_fields=["age"])
        
        assert result.is_valid is False
        assert "age: Invalid number format" in result.errors[0]


class TestCustomValidators:
    """Test cases for custom validators"""
    
    def test_custom_string_validator(self):
        """Test custom string validator"""
        def custom_validator(value):
            return len(value) >= 5
        
        result = input_validator.validate_string(
            "test",
            custom_validators=[custom_validator]
        )
        assert result.is_valid is False
        assert "Custom validation failed" in result.errors[0]
        
        result = input_validator.validate_string(
            "valid_string",
            custom_validators=[custom_validator]
        )
        assert result.is_valid is True
    
    def test_custom_validator_exception(self):
        """Test custom validator with exception"""
        def failing_validator(value):
            raise Exception("Custom validator error")
        
        result = input_validator.validate_string(
            "test",
            custom_validators=[failing_validator]
        )
        assert result.is_valid is False
        assert "Custom validation error" in result.errors[0]
