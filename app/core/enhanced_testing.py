"""
Enhanced Testing Framework
Comprehensive testing utilities, fixtures, and test data management
"""

import asyncio
import pytest
import time
import json
import random
import string
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Test type classifications"""
    UNIT = "unit"
    INTEGRATION = "integration"
    API = "api"
    PERFORMANCE = "performance"
    SECURITY = "security"
    END_TO_END = "end_to_end"


class TestPriority(Enum):
    """Test priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TestResult:
    """Test result information"""
    test_name: str
    test_type: TestType
    priority: TestPriority
    status: str  # passed, failed, skipped, error
    duration: float
    timestamp: datetime
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    coverage_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestData:
    """Test data structure"""
    name: str
    data: Any
    test_type: TestType
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class EnhancedTestManager:
    """Enhanced test management system with comprehensive utilities"""

    def __init__(self):
        self.test_results: List[TestResult] = []
        self.test_data: Dict[str, TestData] = {}
        self.performance_baselines: Dict[str, float] = {}
        self.test_coverage: Dict[str, float] = {}
        self.test_statistics: Dict[str, Any] = defaultdict(int)
        
        # Test configuration
        self.performance_thresholds = {
            "api_response_time": 500.0,  # ms
            "database_query_time": 100.0,  # ms
            "memory_usage": 100.0,  # MB
            "cpu_usage": 80.0  # %
        }

    def generate_test_user(self, user_type: str = "standard") -> Dict[str, Any]:
        """Generate test user data"""
        base_data = {
            "username": f"testuser_{random.randint(1000, 9999)}",
            "email": f"test_{random.randint(1000, 9999)}@example.com",
            "password": "TestPassword123!",
            "first_name": "Test",
            "last_name": "User",
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True,
            "is_verified": True
        }
        
        if user_type == "admin":
            base_data.update({
                "role": "admin",
                "permissions": ["read", "write", "admin", "delete"]
            })
        elif user_type == "premium":
            base_data.update({
                "role": "premium",
                "permissions": ["read", "write", "premium"],
                "subscription": "premium"
            })
        else:
            base_data.update({
                "role": "user",
                "permissions": ["read", "write"]
            })
        
        return base_data

    def generate_test_market(self, market_type: str = "binary") -> Dict[str, Any]:
        """Generate test market data"""
        base_data = {
            "title": f"Test Market {random.randint(1000, 9999)}",
            "description": f"This is a test market for {market_type} trading",
            "category": random.choice(["politics", "sports", "economics", "technology"]),
            "created_by": f"testuser_{random.randint(1000, 9999)}",
            "created_at": datetime.utcnow().isoformat(),
            "end_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
            "status": "active",
            "total_volume": random.randint(1000, 10000),
            "participant_count": random.randint(10, 100)
        }
        
        if market_type == "binary":
            base_data.update({
                "outcome_a": "Yes",
                "outcome_b": "No",
                "current_price_a": round(random.uniform(0.3, 0.7), 3),
                "current_price_b": round(1 - base_data["current_price_a"], 3)
            })
        elif market_type == "multiple":
            base_data.update({
                "outcomes": ["Option A", "Option B", "Option C", "Option D"],
                "current_prices": [round(random.uniform(0.1, 0.4), 3) for _ in range(4)]
            })
        
        return base_data

    def generate_test_trade(self, market_id: str = None) -> Dict[str, Any]:
        """Generate test trade data"""
        return {
            "market_id": market_id or f"market_{random.randint(1000, 9999)}",
            "user_id": f"testuser_{random.randint(1000, 9999)}",
            "outcome": random.choice(["outcome_a", "outcome_b"]),
            "shares": round(random.uniform(1, 100), 2),
            "price": round(random.uniform(0.1, 0.9), 3),
            "total_cost": round(random.uniform(10, 1000), 2),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed",
            "trade_type": random.choice(["buy", "sell"])
        }

    def generate_test_order(self, market_id: str = None) -> Dict[str, Any]:
        """Generate test order data"""
        return {
            "market_id": market_id or f"market_{random.randint(1000, 9999)}",
            "user_id": f"testuser_{random.randint(1000, 9999)}",
            "order_type": random.choice(["limit", "market", "stop"]),
            "side": random.choice(["buy", "sell"]),
            "outcome": random.choice(["outcome_a", "outcome_b"]),
            "shares": round(random.uniform(1, 100), 2),
            "price": round(random.uniform(0.1, 0.9), 3),
            "status": random.choice(["pending", "filled", "cancelled"]),
            "created_at": datetime.utcnow().isoformat(),
            "time_in_force": random.choice(["GTC", "IOC", "FOK"])
        }

    def create_test_database(self) -> str:
        """Create temporary test database"""
        # This would create a temporary database for testing
        # For now, return a mock database URL
        return "sqlite:///test_opinion_market.db"

    def cleanup_test_database(self, db_url: str):
        """Clean up test database"""
        # This would clean up the temporary database
        logger.info(f"Cleaning up test database: {db_url}")

    def measure_performance(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Measure function performance"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            return {
                "execution_time": (end_time - start_time) * 1000,  # ms
                "memory_usage": end_memory - start_memory,  # MB
                "success": True
            }
        except Exception as e:
            end_time = time.time()
            return {
                "execution_time": (end_time - start_time) * 1000,
                "memory_usage": 0,
                "success": False,
                "error": str(e)
            }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def run_performance_test(self, test_name: str, func: Callable, *args, **kwargs) -> TestResult:
        """Run performance test and compare against baseline"""
        start_time = time.time()
        
        try:
            # Measure performance
            metrics = self.measure_performance(func, *args, **kwargs)
            
            # Check against thresholds
            performance_issues = []
            if metrics["execution_time"] > self.performance_thresholds.get("api_response_time", 500):
                performance_issues.append(f"Slow execution: {metrics['execution_time']:.2f}ms")
            
            if metrics["memory_usage"] > self.performance_thresholds.get("memory_usage", 100):
                performance_issues.append(f"High memory usage: {metrics['memory_usage']:.2f}MB")
            
            # Compare with baseline
            baseline = self.performance_baselines.get(test_name)
            if baseline:
                performance_ratio = metrics["execution_time"] / baseline
                if performance_ratio > 1.5:  # 50% slower than baseline
                    performance_issues.append(f"Performance regression: {performance_ratio:.2f}x slower than baseline")
            
            status = "passed" if not performance_issues else "failed"
            error_message = "; ".join(performance_issues) if performance_issues else None
            
            result = TestResult(
                test_name=test_name,
                test_type=TestType.PERFORMANCE,
                priority=TestPriority.HIGH,
                status=status,
                duration=time.time() - start_time,
                timestamp=datetime.utcnow(),
                error_message=error_message,
                performance_metrics=metrics
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                test_type=TestType.PERFORMANCE,
                priority=TestPriority.HIGH,
                status="error",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
            self.test_results.append(result)
            return result

    def run_security_test(self, test_name: str, test_func: Callable, *args, **kwargs) -> TestResult:
        """Run security test"""
        start_time = time.time()
        
        try:
            # Run security test
            result = test_func(*args, **kwargs)
            
            status = "passed" if result else "failed"
            error_message = None if result else "Security test failed"
            
            test_result = TestResult(
                test_name=test_name,
                test_type=TestType.SECURITY,
                priority=TestPriority.CRITICAL,
                status=status,
                duration=time.time() - start_time,
                timestamp=datetime.utcnow(),
                error_message=error_message
            )
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                test_type=TestType.SECURITY,
                priority=TestPriority.CRITICAL,
                status="error",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
            self.test_results.append(result)
            return result

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "passed"])
        failed_tests = len([r for r in self.test_results if r.status == "failed"])
        error_tests = len([r for r in self.test_results if r.status == "error"])
        skipped_tests = len([r for r in self.test_results if r.status == "skipped"])
        
        # Calculate coverage
        total_coverage = sum(self.test_coverage.values()) / len(self.test_coverage) if self.test_coverage else 0
        
        # Performance summary
        performance_tests = [r for r in self.test_results if r.test_type == TestType.PERFORMANCE]
        avg_performance = sum(r.performance_metrics.get("execution_time", 0) for r in performance_tests) / len(performance_tests) if performance_tests else 0
        
        # Security summary
        security_tests = [r for r in self.test_results if r.test_type == TestType.SECURITY]
        security_passed = len([r for r in security_tests if r.status == "passed"])
        security_total = len(security_tests)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "coverage": {
                "total_coverage": total_coverage,
                "by_module": self.test_coverage
            },
            "performance": {
                "average_response_time": avg_performance,
                "performance_tests_count": len(performance_tests),
                "baselines": self.performance_baselines
            },
            "security": {
                "security_tests_passed": security_passed,
                "security_tests_total": security_total,
                "security_score": (security_passed / security_total * 100) if security_total > 0 else 0
            },
            "test_types": {
                test_type.value: len([r for r in self.test_results if r.test_type == test_type])
                for test_type in TestType
            },
            "failed_tests": [
                {
                    "name": r.test_name,
                    "type": r.test_type.value,
                    "error": r.error_message,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.test_results if r.status in ["failed", "error"]
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return report

    def save_test_data(self, name: str, data: Any, test_type: TestType, expires_in_hours: int = 24):
        """Save test data for reuse"""
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours) if expires_in_hours > 0 else None
        
        test_data = TestData(
            name=name,
            data=data,
            test_type=test_type,
            expires_at=expires_at
        )
        
        self.test_data[name] = test_data
        logger.info(f"Test data saved: {name}")

    def get_test_data(self, name: str) -> Optional[Any]:
        """Get saved test data"""
        if name not in self.test_data:
            return None
        
        test_data = self.test_data[name]
        
        # Check if expired
        if test_data.expires_at and test_data.expires_at < datetime.utcnow():
            del self.test_data[name]
            return None
        
        return test_data.data

    def cleanup_expired_data(self):
        """Clean up expired test data"""
        now = datetime.utcnow()
        expired_keys = [
            name for name, data in self.test_data.items()
            if data.expires_at and data.expires_at < now
        ]
        
        for key in expired_keys:
            del self.test_data[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired test data entries")


# Global test manager instance
enhanced_test_manager = EnhancedTestManager()


# Test fixtures and decorators
@pytest.fixture
def test_manager():
    """Provide test manager instance"""
    return enhanced_test_manager


@pytest.fixture
def test_user():
    """Provide test user data"""
    return enhanced_test_manager.generate_test_user()


@pytest.fixture
def test_admin_user():
    """Provide test admin user data"""
    return enhanced_test_manager.generate_test_user("admin")


@pytest.fixture
def test_market():
    """Provide test market data"""
    return enhanced_test_manager.generate_test_market()


@pytest.fixture
def test_trade():
    """Provide test trade data"""
    return enhanced_test_manager.generate_test_trade()


@pytest.fixture
def test_order():
    """Provide test order data"""
    return enhanced_test_manager.generate_test_order()


@pytest.fixture
def test_database():
    """Provide test database"""
    db_url = enhanced_test_manager.create_test_database()
    yield db_url
    enhanced_test_manager.cleanup_test_database(db_url)


def performance_test(test_name: str, baseline_ms: float = None):
    """Decorator for performance testing"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if baseline_ms:
                enhanced_test_manager.performance_baselines[test_name] = baseline_ms
            
            result = enhanced_test_manager.run_performance_test(test_name, func, *args, **kwargs)
            
            if result.status == "failed":
                pytest.fail(f"Performance test failed: {result.error_message}")
            
            return result
        
        return wrapper
    return decorator


def security_test(test_name: str):
    """Decorator for security testing"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = enhanced_test_manager.run_security_test(test_name, func, *args, **kwargs)
            
            if result.status == "failed":
                pytest.fail(f"Security test failed: {result.error_message}")
            
            return result
        
        return wrapper
    return decorator


def integration_test(test_name: str):
    """Decorator for integration testing"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                test_result = TestResult(
                    test_name=test_name,
                    test_type=TestType.INTEGRATION,
                    priority=TestPriority.MEDIUM,
                    status="passed",
                    duration=time.time() - start_time,
                    timestamp=datetime.utcnow()
                )
                
                enhanced_test_manager.test_results.append(test_result)
                return result
                
            except Exception as e:
                test_result = TestResult(
                    test_name=test_name,
                    test_type=TestType.INTEGRATION,
                    priority=TestPriority.MEDIUM,
                    status="failed",
                    duration=time.time() - start_time,
                    timestamp=datetime.utcnow(),
                    error_message=str(e)
                )
                
                enhanced_test_manager.test_results.append(test_result)
                raise
        
        return wrapper
    return decorator
